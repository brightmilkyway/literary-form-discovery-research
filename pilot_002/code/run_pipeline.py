"""Run the frozen Pilot 002 discovery pipeline."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import common as C
import geometry as G
import representations as R


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representations", nargs="+", default=list(C.REPRESENTATIONS),
        choices=list(C.REPRESENTATIONS),
        help="Subset for staged execution; the frozen full run uses all five.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute matrices even when cached.")
    return parser.parse_args()


def matrix_path(scheme: str, representation: str):
    return C.MATRIX_DIR / f"{scheme}__{representation}.npz"


def save_matrix(path, S, P, metadata):
    np.savez_compressed(path, similarity=S.astype(np.float32),
                        percentile=P.astype(np.float32),
                        metadata=np.array(json.dumps(metadata,
                                                     ensure_ascii=False)))


def load_matrix(path):
    data = np.load(path, allow_pickle=False)
    return data["similarity"], data["percentile"], json.loads(
        str(data["metadata"].item()))


def main():
    args = parse_args()
    started = time.time()
    body, title_offset = C.load_body()
    partitions = C.build_partitions(body)
    checks = C.validate_partitions(body, partitions)
    if not all(row["ok"] for row in checks):
        raise RuntimeError(f"coordinate validation failed: {checks}")

    units_payload = {
        "source": str(C.SOURCE), "source_sha256": C.source_fingerprint(),
        "title_offset": title_offset, "body_characters": len(body),
        "body_words": len(C.word_tokens(body)), "schemes": {
            scheme: [unit.record() for unit in units]
            for scheme, units in partitions.items()
        },
    }
    C.save_json(units_payload, C.DERIVED_DIR / "units.json")
    C.write_csv(C.TABLE_DIR / "partition_validation.csv", checks)

    neural = {}
    all_events = []
    matrix_rows = []
    total_words = len(C.word_tokens(body))

    for representation in args.representations:
        if representation.endswith("_white"):
            print(f"loading neural observer: {representation}", flush=True)
            neural[representation] = R.NeuralWhitened(representation)
        for scheme, units in partitions.items():
            path = matrix_path(scheme, representation)
            t0 = time.time()
            if path.exists() and not args.force:
                S, P, metadata = load_matrix(path)
                events, _, geometry_meta = G.extract_all(
                    S, units, scheme, representation, total_words)
                metadata.update(geometry_meta)
                source = "cache"
            else:
                if representation.endswith("_white"):
                    S, rep_meta = neural[representation].matrix(scheme, units)
                else:
                    S, rep_meta = R.lexical_matrix(representation, units)
                if S.shape != (len(units), len(units)) or not np.isfinite(S).all():
                    raise RuntimeError(f"invalid matrix {scheme}/{representation}")
                events, P, geometry_meta = G.extract_all(
                    S, units, scheme, representation, total_words)
                metadata = {"scheme": scheme, "scale": units[0].scale,
                            "phase": units[0].phase,
                            "representation": representation,
                            "n_units": len(units), **rep_meta, **geometry_meta}
                save_matrix(path, S, P, metadata)
                source = "computed"
            counts = {kind: sum(e["geometry"] == kind for e in events)
                      for kind in ("point", "stripe", "block", "valley")}
            matrix_rows.append({
                "scheme": scheme, "scale": units[0].scale,
                "phase": units[0].phase, "representation": representation,
                "n_units": len(units), **counts,
                "eligible_remote_pairs": metadata["eligible_remote_pairs"],
                "seconds": round(time.time() - t0, 3), "source": source,
                "diagnostic": json.dumps(metadata, ensure_ascii=False,
                                           sort_keys=True),
            })
            all_events.extend(events)
            print(f"{representation:14s} {scheme:12s} "
                  f"n={len(units):4d} events={len(events):5d} "
                  f"{time.time()-t0:6.1f}s {source}", flush=True)

    C.write_csv(C.TABLE_DIR / "matrix_inventory.csv", matrix_rows)
    C.write_csv(C.TABLE_DIR / "geometry_events.csv", all_events)

    candidates = G.aggregate(all_events, total_words, body)
    C.write_csv(C.TABLE_DIR / "candidates.csv", candidates)

    run_meta = {
        "protocol": "protocol.md (frozen before first run)",
        "source_sha256": C.source_fingerprint(),
        "representations_requested": args.representations,
        "complete_five_representation_run": set(args.representations) ==
        set(C.REPRESENTATIONS),
        "schemes": len(partitions), "matrices": len(matrix_rows),
        "events": len(all_events), "candidates": len(candidates),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    C.save_json(run_meta, C.DERIVED_DIR / "run_metadata.json")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

