"""Streaming corpus readers (plain text, .gz, .xz, .bz2)."""
from __future__ import annotations

import bz2
import gzip
import lzma
import os
from typing import Iterable, Iterator, Union

PathLike = Union[str, "os.PathLike[str]"]


def _open(path: PathLike, encoding: str = "utf-8"):
    p = str(path)
    if p.endswith(".xz"):
        return lzma.open(p, "rt", encoding=encoding, errors="replace")
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding=encoding, errors="replace")
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding=encoding, errors="replace")
    return open(p, "rt", encoding=encoding, errors="replace")


def _as_list(paths: Union[PathLike, Iterable[PathLike]]):
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)


def iter_lines(
    paths: Union[PathLike, Iterable[PathLike]], encoding: str = "utf-8", max_chars: int = 1_000_000
) -> Iterator[str]:
    """Yield non-empty, stripped lines from one or more files.

    A line longer than ``max_chars`` is yielded in pieces cut at whitespace, so a file without
    line breaks (GPAC is one 1 GB line) is read in bounded memory.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    for path in _as_list(paths):
        with _open(path, encoding) as fh:
            carry = ""  # unfinished line text, always shorter than max_chars
            while True:
                chunk = fh.readline(max_chars - len(carry))
                if not chunk:
                    break
                text = carry + chunk
                if text.endswith("\n") or len(text) < max_chars:  # a whole line, or the end of the file
                    carry = ""
                    text = text.strip()
                    if text:
                        yield text
                    continue
                # the line continues past max_chars: yield up to its last whitespace
                cut = max(text.rfind(" "), text.rfind("\t"))
                if cut <= 0:
                    cut = len(text)  # no whitespace in max_chars characters: cut anyway
                piece, carry = text[:cut].strip(), text[cut:]
                if piece:
                    yield piece
            carry = carry.strip()
            if carry:
                yield carry


def iter_documents(paths: Union[PathLike, Iterable[PathLike]], encoding: str = "utf-8") -> Iterator[str]:
    """Yield documents separated by blank lines (the CC-100 layout)."""
    for path in _as_list(paths):
        buf = []
        with _open(path, encoding) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    buf.append(line)
                elif buf:
                    yield "\n".join(buf)
                    buf = []
        if buf:
            yield "\n".join(buf)
