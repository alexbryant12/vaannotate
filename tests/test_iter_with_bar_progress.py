from __future__ import annotations

from pathlib import Path
from typing import List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaannotate.vaannotate_ai_backend import engine


class SimpleLogCollector:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self._progress_active = False
        self._progress_text = ""

    def append(self, message: str) -> None:
        if not message:
            return
        if message.startswith("\r"):
            text = message[1:].strip()
            if not text:
                return
            if not self._progress_active:
                self.lines.append(text)
                self._progress_active = True
            else:
                if self.lines:
                    self.lines[-1] = text
                else:
                    self.lines.append(text)
            self._progress_text = text
            return

        clean = message.strip()
        if not clean:
            return
        if self._progress_active:
            if clean == self._progress_text:
                self._progress_active = False
                self._progress_text = clean
                return
            self._progress_active = False
        self.lines.append(clean)
        self._progress_text = ""


class FakeTTY:
    def __init__(self, collector: SimpleLogCollector | None = None) -> None:
        self.collector = collector
        self.writes: List[str] = []

    def write(self, data: str) -> int:
        self.writes.append(data)
        if self.collector is not None:
            for chunk in split_stream_chunks(data):
                if chunk:
                    self.collector.append(chunk)
        return len(data)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True


def split_stream_chunks(data: str) -> List[str]:
    chunks: List[str] = []
    buffer = ""
    for ch in data:
        if ch == "\r":
            if buffer:
                chunks.append(buffer)
            buffer = "\r"
        else:
            buffer += ch
            if ch == "\n":
                chunks.append(buffer)
                buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def test_iter_with_bar_tty_emits_without_escape_sequences(monkeypatch):
    fake = FakeTTY()
    monkeypatch.setattr(engine._sys, "stderr", fake)

    list(
        engine.iter_with_bar(
            "Testing",
            range(3),
            total=3,
            min_interval_s=0.0,
            ascii_only=True,
            logger=engine.LOGGER,
        )
    )

    output = "".join(fake.writes)
    assert "\x1b" not in output
    assert output.count("\r") >= 2
    last_line = output.split("\r")[-1].strip()
    assert last_line.startswith("Testing")
    assert "100%  3/3" in last_line


def test_iter_with_bar_progress_keeps_logs(monkeypatch):
    collector = SimpleLogCollector()
    fake_tty = FakeTTY(collector)
    monkeypatch.setattr(engine._sys, "stderr", fake_tty)

    collector.append("Log before 1")
    collector.append("Log before 2")

    list(
        engine.iter_with_bar(
            "Embedding",
            range(3),
            total=3,
            min_interval_s=0.0,
            ascii_only=True,
            logger=engine.LOGGER,
        )
    )

    collector.append("Log after")

    assert len(collector.lines) == 4
    assert collector.lines[0] == "Log before 1"
    assert collector.lines[1] == "Log before 2"
    assert "Embedding" in collector.lines[2]
    assert collector.lines[3] == "Log after"
