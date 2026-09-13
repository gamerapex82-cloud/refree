"""Phase 4 — real NASDAQ ITCH tape smoke (plan_2.md Phase 4, work package §4.2).

Two layers:

* **Always on** (no network, no data): ``fetch_itch.py``'s framing/slicing
  logic is exercised on a hand-built length-prefixed ITCH stream that mixes
  ``S`` / ``R`` / order messages for two symbols — the per-symbol slice must
  contain exactly the session events, the symbol's directory record, and
  that symbol's order messages, byte for byte.
* **Real bytes** (skipped unless ``data/itch/<day>/<SYMBOL>.itch`` exists —
  produce it with ``python python_quant/scripts/fetch_itch.py``; ``data/`` is
  gitignored and never committed): the parser consumes the real file with
  **0 truncated** messages, the replay is integrity-clean, the order-level
  tracker sees no unknown ids, and the prefix runs in bounded time.

Measured on 12/30/2019 AAPL (full day, 1.52 M messages): 0 truncated, 1
integrity flag (``empty_bbo`` on the very first pre-market message — an empty
book, not a decode error), tracker 0 unknown ids, and Engine-vs-Stub ladder
parity exact over the pre-market prefix (7,037 frames) — see ``progress_b.md``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_port import StubBookAdapter
from nexus_quant.book_state import Side
from nexus_quant.itch_parser import (
    EventType,
    ItchParseStats,
    NormalizedEvent,
    encode_event,
    iter_itch_events,
)
from nexus_quant.replay import ReplayEngine
from nexus_quant.research.queue_dynamics import OrderLevelTracker

_ROOT = Path(__file__).resolve().parents[2]
_DAY = os.environ.get("NEXUS_ITCH_DAY", "12302019")
_SYMBOL = os.environ.get("NEXUS_ITCH_SYMBOL", "AAPL")
_TAPE = _ROOT / "data" / "itch" / _DAY / f"{_SYMBOL}.itch"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_itch", _ROOT / "python_quant" / "scripts" / "fetch_itch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _framed(body: bytes) -> bytes:
    return len(body).to_bytes(2, "big") + body


def _system_event(code: bytes, ts: int) -> bytes:
    # S: locate(2) tracking(2) ts(6) event(1)  -> 12 bytes incl. type
    return _framed(b"S" + (0).to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big") + code)


def _directory(locate: int, stock: str, ts: int) -> bytes:
    # R: locate(2) tracking(2) ts(6) stock(8) + 20 more bytes of attributes -> 39 incl. type
    body = (b"R" + locate.to_bytes(2, "big") + (0).to_bytes(2, "big") + ts.to_bytes(6, "big")
            + stock.ljust(8).encode() + b"Q" + b"N" + (100).to_bytes(4, "big") + b"N" + b"Q" + b"N" + b"  "
            + b"N" + b"N" + b" " + b"N" + (0).to_bytes(4, "big") + b"N")
    assert len(body) == 39, len(body)
    return _framed(body)


def _add(locate: int, oid: int, side: Side, px: int, sz: int, ts: int) -> bytes:
    ev = NormalizedEvent(EventType.ADD, ts, oid, side, px, sz)
    raw = bytearray(encode_event(ev, framed=False))
    raw[1:3] = locate.to_bytes(2, "big")  # encode_event hard-codes locate=1
    return _framed(bytes(raw))


def _delete(locate: int, oid: int, ts: int) -> bytes:
    raw = bytearray(encode_event(NormalizedEvent(EventType.DELETE, ts, oid, Side.NONE, 0, 0), framed=False))
    raw[1:3] = locate.to_bytes(2, "big")
    return _framed(bytes(raw))


def test_symbol_slicer_keeps_exactly_the_symbol_stream(tmp_path: Path) -> None:
    fetch = _load_fetch_module()
    ts = 34_200_000_000_000
    stream = b"".join([
        _system_event(b"O", ts - 5),
        _directory(7, "AAPL", ts - 4),
        _directory(9, "QQQ", ts - 4),
        _system_event(b"Q", ts - 1),
        _add(7, 11, Side.Bid, 2_890_000, 100, ts + 1),
        _add(9, 21, Side.Ask, 2_130_000, 50, ts + 2),
        _add(7, 12, Side.Ask, 2_890_500, 40, ts + 3),
        _delete(9, 21, ts + 4),
        _delete(7, 11, ts + 5),
        _system_event(b"M", ts + 10),
    ])
    slicer = fetch._SymbolSlicer(tmp_path, {"AAPL"})
    # feed in awkward chunk boundaries to exercise the framing buffer
    prev = 0
    for cut in (5, 17, 40, 41, 90, 150, len(stream)):
        slicer.feed(stream[prev:cut])
        prev = cut
    slicer.close()
    assert slicer.messages == 10 and slicer.locate_to_symbol == {7: "AAPL"}
    assert slicer.counts["AAPL"] == {"R": 1, "A": 2, "D": 1}
    out = (tmp_path / "AAPL.itch").read_bytes()
    expected = b"".join([
        _system_event(b"O", ts - 5),  # replayed S messages seen before the R
        _directory(7, "AAPL", ts - 4),
        _system_event(b"Q", ts - 1),
        _add(7, 11, Side.Bid, 2_890_000, 100, ts + 1),
        _add(7, 12, Side.Ask, 2_890_500, 40, ts + 3),
        _delete(7, 11, ts + 5),
        _system_event(b"M", ts + 10),
    ])
    assert out == expected
    # and the slice is a valid tape for the parser + replay + tracker
    stats = ItchParseStats()
    events = list(iter_itch_events(out, stats=stats))
    assert stats.truncated == 0 and [e.kind for e in events] == [EventType.ADD, EventType.ADD, EventType.DELETE]
    rep = ReplayEngine(events, StubBookAdapter())
    last = rep.step_n(3)
    assert last is not None and rep.applied == 3 and last.issues == []
    assert int(last.state["ask_px"][0]) == 2_890_500 and int(last.state["bid_px"][0]) == 0
    assert (tmp_path / "QQQ.itch").exists() is False


def test_tape_url_and_manifest_shape() -> None:
    fetch = _load_fetch_module()
    assert fetch.tape_url("12302019") == "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/12302019.NASDAQ_ITCH50.gz"
    assert fetch.DEFAULT_SYMBOLS and fetch.DEFAULT_DAY == "12302019"


@pytest.mark.skipif(not _TAPE.exists(), reason=f"real tape not fetched: {_TAPE} (run scripts/fetch_itch.py)")
def test_real_tape_parses_clean() -> None:
    """Parser 0 truncated · replay integrity clean (bar the empty pre-open book) ·
    tracker 0 unknown ids · bounded runtime on the first 150k events."""
    stats = ItchParseStats()
    t0 = time.time()
    events = []
    for ev in iter_itch_events(_TAPE, stats=stats):
        events.append(ev)
        if len(events) >= 150_000:
            break
    assert stats.truncated == 0
    assert len(events) >= 1_000
    assert all(e.price_ticks >= 0 and e.size >= 0 for e in events)
    assert all(events[i].ts_ns <= events[i + 1].ts_ns for i in range(len(events) - 1))  # tape order
    rep = ReplayEngine(events, StubBookAdapter())
    tracker = OrderLevelTracker()
    issue_codes: dict[str, int] = {}
    for frame in rep.frames():
        tracker.on_event(frame.event)
        for iss in frame.issues:
            issue_codes[iss.code] = issue_codes.get(iss.code, 0) + 1
    assert rep.skipped == 0, "every real event must apply to the book"
    assert set(issue_codes) <= {"empty_bbo"}, issue_codes  # never crossed / locked / negative / unsorted
    assert tracker.stats["unknown_id"] == 0
    assert time.time() - t0 < 120.0
