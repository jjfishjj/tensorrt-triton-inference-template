"""Rendering results as a table and as JSON.

The table is the deliverable. Unavailable backends keep their row and carry
their reason, because a comparison that silently drops what it could not run is
how four invented numbers end up in a slide.
"""
from __future__ import annotations

import json
from pathlib import Path


def table(results, accuracies: dict, host: dict, reference: str) -> str:
    acc = {a.backend: a for a in accuracies.values()}
    w = 22
    lines = [
        f"host: {host['processor']} · {host['platform']}",
        "",
        f"{'backend':<{w}} {'p50 ms':>8} {'p99 ms':>8} {'inf/s':>9} "
        f"{'max |Δ|':>10} {'top-1':>7}",
        "-" * (w + 46),
    ]
    for r in results:
        if not r.available:
            lines.append(f"{r.backend:<{w}} {'—':>8} {'—':>8} {'—':>9}   "
                         f"unavailable: {r.reason}")
            continue
        if r.error:
            lines.append(f"{r.backend:<{w}} {'—':>8} {'—':>8} {'—':>9}   "
                         f"error: {r.error}")
            continue
        s = r.latency
        a = acc.get(r.backend)
        adiff = f"{a.max_abs_diff:.2e}" if a else ("reference" if r.backend == reference else "—")
        top1 = f"{a.top1_agreement * 100:.1f}%" if a else ("—" if r.backend != reference else "—")
        lines.append(f"{r.backend:<{w}} {s.p50_ms:8.2f} {s.p99_ms:8.2f} "
                     f"{s.throughput_ips:9.1f} {adiff:>10} {top1:>7}")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
