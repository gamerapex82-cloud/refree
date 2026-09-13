#!/usr/bin/env python3
"""Fetch a public NASDAQ TotalView-ITCH 5.0 sample day and slice it per symbol.

Source / provenance
-------------------
NASDAQ publishes full-day ITCH 5.0 sample files for evaluation at

    https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/<MMDDYYYY>.NASDAQ_ITCH50.gz

(e.g. ``12302019.NASDAQ_ITCH50.gz``, ~3.5 GB gzipped, ~10 GB raw, ~300 M
messages). The files are 2-byte big-endian length-prefixed ITCH 5.0 messages.

License note: the sample files are provided by Nasdaq for evaluation of the
TotalView-ITCH product. They are **not** redistributed by this repository —
``data/`` and ``*.itch`` are gitignored and every artifact this script writes
carries a ``manifest.json`` with the exact source URL, byte range, and SHA-256
so a run is reproducible from the public source. Do not commit the output.

What this script does
---------------------
It streams the gzip over HTTP (optionally a bounded ``Range`` prefix for smoke
tests), inflates it on the fly, frames every message, and writes **one framed
``.itch`` file per requested symbol** containing:

* every ``S`` (System Event) message — session boundaries for all symbols;
* the symbol's ``R`` (Stock Directory) message — locate → symbol proof;
* every order/print message for that locate code: ``A F E C X D U P``.

Everything else (other symbols, NOII, cross, admin) is dropped, which is what
turns a 10 GB day into a few tens of MB per liquid symbol that
``iter_itch_events`` / ``ReplayEngine`` consume in seconds.

Run (from repo root)::

    python python_quant/scripts/fetch_itch.py --day 12302019 --symbols QQQ,AAPL
    # smoke: only the first 64 MB of the gzip (pre-market + directory)
    python python_quant/scripts/fetch_itch.py --day 12302019 --symbols QQQ --max-gz-bytes 67108864

Output: ``data/itch/<day>/<SYMBOL>.itch`` + ``data/itch/<day>/manifest.json``.
No third-party dependency — stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zlib
from pathlib import Path

DEFAULT_BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"
DEFAULT_DAY = "12302019"  # smallest recent full day on the public server
DEFAULT_SYMBOLS = ("QQQ", "AAPL")

# Message types carried over into every per-symbol slice.
_ORDER_TYPES = frozenset(b"AFECXDUP")
_SESSION_TYPE = ord("S")
_DIRECTORY_TYPE = ord("R")


def tape_url(day: str, base: str = DEFAULT_BASE) -> str:
    return f"{base}{day}.NASDAQ_ITCH50.gz"


class _SymbolSlicer:
    """Frames a raw ITCH byte stream and fans messages out per locate code."""

    def __init__(self, out_dir: Path, symbols: set[str]) -> None:
        self.out_dir = out_dir
        self.wanted = {s.upper() for s in symbols}
        self.locate_to_symbol: dict[int, str] = {}
        self.files: dict[int, object] = {}
        self.counts: dict[str, dict[str, int]] = {s: {} for s in self.wanted}
        self.session_msgs: list[bytes] = []  # S messages seen so far (replayed into late-opened files)
        self.messages = 0
        self.raw_bytes = 0
        self.last_ts_ns = 0
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        self.raw_bytes += len(chunk)
        buf = self._buf
        n = len(buf)
        i = 0
        while i + 2 <= n:
            ln = (buf[i] << 8) | buf[i + 1]
            end = i + 2 + ln
            if end > n:
                break
            typ = buf[i + 2]
            locate = (buf[i + 3] << 8) | buf[i + 4]
            self.messages += 1
            if typ == _SESSION_TYPE:
                msg = bytes(buf[i:end])
                self.last_ts_ns = int.from_bytes(msg[7:13], "big")
                self.session_msgs.append(msg)
                for fh in self.files.values():
                    fh.write(msg)  # type: ignore[attr-defined]
            elif typ == _DIRECTORY_TYPE:
                stock = bytes(buf[i + 13 : i + 21]).decode("ascii", "replace").strip()
                if stock in self.wanted and locate not in self.files:
                    self.locate_to_symbol[locate] = stock
                    fh = open(self.out_dir / f"{stock}.itch", "wb")  # noqa: SIM115 — long-lived handle
                    fh.writelines(self.session_msgs)
                    fh.write(bytes(buf[i:end]))
                    self.files[locate] = fh
                    self.counts[stock]["R"] = 1
            elif typ in _ORDER_TYPES and locate in self.files:
                self.files[locate].write(bytes(buf[i:end]))  # type: ignore[attr-defined]
                c = self.counts[self.locate_to_symbol[locate]]
                key = chr(typ)
                c[key] = c.get(key, 0) + 1
            i = end
        del buf[:i]

    def close(self) -> None:
        for fh in self.files.values():
            fh.close()  # type: ignore[attr-defined]
        self.files.clear()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(
    *,
    day: str,
    symbols: set[str],
    out_root: Path,
    base: str = DEFAULT_BASE,
    max_gz_bytes: int | None = None,
    chunk: int = 1 << 20,
    log=print,
) -> dict:
    """Stream ``day`` from ``base``, slice ``symbols`` into ``out_root/day/``."""
    url = tape_url(day, base)
    out_dir = out_root / day
    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "nexus-lob-fetch-itch/1.0"})
    if max_gz_bytes is not None:
        req.add_header("Range", f"bytes=0-{int(max_gz_bytes) - 1}")

    slicer = _SymbolSlicer(out_dir, symbols)
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    gz_bytes = 0
    t0 = time.time()
    next_log = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            gz_bytes += len(block)
            slicer.feed(inflater.decompress(block))
            if gz_bytes >= next_log:
                hours = slicer.last_ts_ns / 3.6e12
                log(
                    f"  {gz_bytes / 1e6:8.1f} MB gz · {slicer.messages / 1e6:7.2f} M msgs · "
                    f"tape clock {hours:5.2f} h · {time.time() - t0:6.1f} s"
                )
                next_log += 256 << 20
        # a Range-truncated gzip has no valid trailer — flush what inflated cleanly
        try:
            slicer.feed(inflater.flush())
        except zlib.error:
            pass
    slicer.close()

    outputs = {}
    for locate, stock in slicer.locate_to_symbol.items():
        p = out_dir / f"{stock}.itch"
        outputs[stock] = {
            "file": p.name,
            "locate": locate,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
            "messages": slicer.counts[stock],
        }
    missing = sorted(slicer.wanted - set(outputs))
    manifest = {
        "source_url": url,
        "day": day,
        "gz_bytes_fetched": gz_bytes,
        "range_limited": max_gz_bytes is not None,
        "raw_bytes_inflated": slicer.raw_bytes,
        "messages_framed": slicer.messages,
        "last_tape_ts_ns": slicer.last_ts_ns,
        "symbols": outputs,
        "symbols_not_found": missing,
        "fetched_at_unix": int(time.time()),
        "license": (
            "Nasdaq TotalView-ITCH sample data, provided by Nasdaq for evaluation; "
            "not redistributed by this repository (data/ is gitignored)."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"done: {gz_bytes / 1e6:.1f} MB gz → {slicer.messages / 1e6:.2f} M msgs in {time.time() - t0:.1f} s")
    for stock, info in outputs.items():
        log(f"  {stock:<6} {info['bytes'] / 1e6:7.2f} MB  {info['messages']}")
    if missing:
        log(f"  not found in directory: {missing}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default=DEFAULT_DAY, help="MMDDYYYY of the public sample (default 12302019)")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated tickers")
    ap.add_argument("--out", type=Path, default=Path("data/itch"), help="output root (gitignored)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="override the public base URL")
    ap.add_argument(
        "--max-gz-bytes", type=int, default=None,
        help="only fetch this many gzip bytes (HTTP Range) — smoke tests / CI",
    )
    args = ap.parse_args(argv)
    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    if not symbols:
        ap.error("--symbols must name at least one ticker")
    print(f"fetching {tape_url(args.day, args.base)} → {args.out / args.day}  symbols={sorted(symbols)}")
    fetch(day=args.day, symbols=symbols, out_root=args.out, base=args.base, max_gz_bytes=args.max_gz_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
