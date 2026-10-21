"""Normalized Compression Distance (Cilibrasi & Vitányi, 2005).

``NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))``

* ``C`` is a *practical* compressor, not the ideal Kolmogorov complexity.  The
  gap between the two is the main reason NCD is sensitive to text length and to
  the compressor chosen; both are measured explicitly here.
* The concatenation order used is the original reading order (x before y),
  because reversing it can change ``C(xy)``.
* Length is a known confound, so we also report the raw compressed sizes and a
  length-corrected view (partial correlation of NCD with position distance,
  controlling for the size difference of the two units).
"""

from __future__ import annotations

import bz2
import lzma
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from common import Unit


# --------------------------------------------------------------------------
# Compressors
# --------------------------------------------------------------------------

def _zlib(x: bytes) -> int:
    return len(zlib.compress(x, 9))


def _bz2(x: bytes) -> int:
    return len(bz2.compress(x, 9))


def _lzma(x: bytes) -> int:
    return len(lzma.compress(x, preset=9 | lzma.PRESET_EXTREME))


COMPRESSORS: Dict[str, Callable[[bytes], int]] = {
    "zlib": _zlib,
    "bz2": _bz2,
    "lzma": _lzma,
}


@dataclass
class NCDResult:
    compressor: str
    units: List[Unit]
    matrix: np.ndarray          # symmetric, 0 on the diagonal
    sizes: np.ndarray           # C(x_i) per unit
    order: str                  # "reading" (x before y)


def ncd_matrix(units: Sequence[Unit], compressor: str = "zlib") -> NCDResult:
    comp = COMPRESSORS[compressor]
    blobs = [u.text.encode("utf-8") for u in units]
    sizes = np.array([comp(b) for b in blobs], dtype=float)
    n = len(units)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            cxy = comp(blobs[i] + blobs[j])          # reading order
            d = (cxy - min(sizes[i], sizes[j])) / max(sizes[i], sizes[j])
            M[i, j] = M[j, i] = d
    return NCDResult(compressor=compressor, units=list(units), matrix=M,
                     sizes=sizes, order="reading")


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------

def non_adjacent_pairs(res: NCDResult, top: int = 15, min_gap: int = 2
                       ) -> List[dict]:
    """Lowest-distance (most similar) pairs that are NOT near neighbours."""
    M = res.matrix
    n = len(res.units)
    pairs = []
    for i in range(n):
        for j in range(i + min_gap, n):
            pairs.append((float(M[i, j]), i, j))
    pairs.sort(key=lambda t: t[0])
    out = []
    for d, i, j in pairs[:top]:
        out.append({
            "i": i, "j": j, "distance": d, "gap": j - i,
            "i_span": [res.units[i].start, res.units[i].end],
            "j_span": [res.units[j].start, res.units[j].end],
            "i_preview": _preview(res.units[i].text),
            "j_preview": _preview(res.units[j].text),
        })
    return out


def _preview(text: str, width: int = 120) -> str:
    import re
    s = re.sub(r"\s+", " ", text).strip()
    return s[:width] + ("…" if len(s) > width else "")


def adjacent_profile(res: NCDResult) -> dict:
    """Mean NCD as a function of index distance (the 'adjacency effect')."""
    M = res.matrix
    n = len(res.units)
    by_gap: Dict[int, List[float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            by_gap.setdefault(j - i, []).append(float(M[i, j]))
    return {g: {"mean": float(np.mean(v)), "n": len(v)}
            for g, v in sorted(by_gap.items())}


def length_confound(res: NCDResult) -> dict:
    """Correlation of NCD with (a) index distance and (b) size difference."""
    M = res.matrix
    n = len(res.units)
    iu, ju = np.triu_indices(n, k=1)
    dist = (ju - iu).astype(float)
    d = M[iu, ju]
    size_diff = np.abs(res.sizes[iu] - res.sizes[ju])
    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "corr_ncd_index_distance": corr(d, dist),
        "corr_ncd_size_difference": corr(d, size_diff),
        "corr_index_distance_size_difference": corr(dist, size_diff),
    }
