"""Prepare prefix-only interpretation requests and validate returned outputs.

No provider is called here. This keeps model choice, cost, credentials, and the
decision to execute outside the local reproducible preparation step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import common as C

SYSTEM_PROMPT = """You are participating in a sequential literary-text experiment. Use only the supplied text prefix as evidence. Do not use or mention remembered later portions of the work. Describe the understanding supported at this point without assigning a predefined theme or reconstructing a complete plot. Return valid JSON with exactly three string fields: current_understanding, salient_uncertainties, and evidence_basis. Write 120-180 English words in total. Treat uncertainty explicitly and do not claim to model a human reader."""

USER_TEMPLATE = """TEXT PREFIX (ends at checkpoint {checkpoint_id}):
---
{prefix}
---
Give the requested prefix-only interpretation."""


def prompt_hash() -> str:
    frozen_prompt = SYSTEM_PROMPT + "\n<USER_TEMPLATE>\n" + USER_TEMPLATE
    return hashlib.sha256(frozen_prompt.encode("utf-8")).hexdigest()


def prepare() -> Path:
    config = C.load_config()
    body, _ = C.load_body()
    paras = C.paragraphs(body)
    checkpoints = C.build_checkpoints(body, paras, config["checkpoint_count"])
    settings = config["interpretive"]
    rows = []
    for checkpoint in checkpoints:
        prefix = body[:checkpoint["char_end"]]
        rows.append({
            "checkpoint_id": checkpoint["checkpoint_id"],
            "status": "prepared_not_run",
            "model": settings["model"],
            "prompt_version": settings["prompt_version"],
            "prompt_sha256": prompt_hash(),
            "parameters": {
                "temperature": settings["temperature"],
                "max_output_tokens": settings["max_output_tokens"],
                "response_format": settings["response_format"],
            },
            "input": {
                "system": SYSTEM_PROMPT,
                "user": USER_TEMPLATE.format(
                    checkpoint_id=checkpoint["checkpoint_id"], prefix=prefix),
            },
            "prefix_sha256": checkpoint["prefix_sha256"],
        })
    path = C.DERIVED_DIR / "interpretation_requests.jsonl"
    C.write_jsonl(path, rows)
    return path


def validate_responses(response_path: Path) -> Path:
    request_path = C.DERIVED_DIR / "interpretation_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError("run the prepare command before validating responses")
    requests = {row["checkpoint_id"]: row for row in C.read_jsonl(request_path)}
    responses = C.read_jsonl(response_path)
    response_ids = [r.get("checkpoint_id") for r in responses]
    if len(response_ids) != len(requests) or len(set(response_ids)) != len(requests) \
            or set(response_ids) != set(requests):
        raise ValueError("responses must contain exactly one row for every checkpoint")
    merged = []
    for response in responses:
        checkpoint_id = response["checkpoint_id"]
        output = response.get("output")
        model = response.get("model")
        request = requests[checkpoint_id]
        if response.get("prompt_sha256") != request["prompt_sha256"] or \
                response.get("prefix_sha256") != request["prefix_sha256"]:
            raise ValueError(f"{checkpoint_id}: request provenance hashes do not match")
        if not isinstance(output, str) or not output.strip() or not model:
            raise ValueError(f"{checkpoint_id}: non-empty output and model are required")
        parsed = json.loads(output)
        required = {"current_understanding", "salient_uncertainties", "evidence_basis"}
        if set(parsed) != required or not all(isinstance(parsed[k], str) for k in required):
            raise ValueError(f"{checkpoint_id}: output does not match the frozen JSON schema")
        merged.append({
            **request,
            "status": "completed",
            "model": model,
            "output": output,
            "provider_metadata": response.get("provider_metadata", {}),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    suffix = response_path.stem.removeprefix("deepseek_responses")
    path = C.DERIVED_DIR / f"interpretation_runs{suffix}.jsonl"
    C.write_jsonl(path, merged)
    return path


def _save_response_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    C.write_jsonl(temporary, rows)
    temporary.replace(path)


def run_deepseek(replicate: int) -> Path:
    """Execute one resumable replicate against the official DeepSeek endpoint."""
    import requests

    config = C.load_config()["interpretive"]
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    request_path = prepare()
    requests_to_run = C.read_jsonl(request_path)
    path = C.DERIVED_DIR / f"deepseek_responses_r{replicate:02d}.jsonl"
    completed = C.read_jsonl(path) if path.exists() else []
    by_id = {row["checkpoint_id"]: row for row in completed}
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    session = requests.Session()
    for position, request_row in enumerate(requests_to_run, 1):
        checkpoint_id = request_row["checkpoint_id"]
        if checkpoint_id in by_id:
            continue
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": request_row["input"]["system"]},
                {"role": "user", "content": request_row["input"]["user"]},
            ],
            "temperature": config["temperature"],
            "max_tokens": config["max_output_tokens"],
            "response_format": {"type": "json_object"},
            "thinking": {"type": config["thinking"]},
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                response = session.post(endpoint, headers=headers, json=payload,
                                        timeout=(20, 180))
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                if not content or choice.get("finish_reason") != "stop":
                    raise RuntimeError(
                        f"empty/truncated response: {choice.get('finish_reason')}")
                parsed = json.loads(content)
                required = {"current_understanding", "salient_uncertainties",
                            "evidence_basis"}
                if set(parsed) != required:
                    raise RuntimeError(f"unexpected JSON keys: {sorted(parsed)}")
                row = {
                    "checkpoint_id": checkpoint_id,
                    "model": data.get("model", config["model"]),
                    "prompt_sha256": request_row["prompt_sha256"],
                    "prefix_sha256": request_row["prefix_sha256"],
                    "output": content,
                    "provider_metadata": {
                        "provider": "deepseek_official",
                        "response_id": data.get("id"),
                        "created": data.get("created"),
                        "system_fingerprint": data.get("system_fingerprint"),
                        "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage", {}),
                        "replicate": replicate,
                    },
                }
                by_id[checkpoint_id] = row
                ordered = [by_id[r["checkpoint_id"]] for r in requests_to_run
                           if r["checkpoint_id"] in by_id]
                _save_response_rows(path, ordered)
                usage = row["provider_metadata"]["usage"]
                print(f"[{position:02d}/{len(requests_to_run)}] {checkpoint_id} "
                      f"prompt={usage.get('prompt_tokens', '?')} "
                      f"completion={usage.get('completion_tokens', '?')}", flush=True)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt < 4:
                    time.sleep(2 ** attempt)
        if last_error is not None:
            raise RuntimeError(f"{checkpoint_id} failed after retries") from last_error
    return validate_responses(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-deepseek")
    run.add_argument("--replicate", type=int, default=1)
    validate = sub.add_parser("validate")
    validate.add_argument("responses", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare()
    elif args.command == "run-deepseek":
        path = run_deepseek(args.replicate)
    else:
        path = validate_responses(args.responses)
    print(path)


if __name__ == "__main__":
    main()
