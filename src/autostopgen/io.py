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


def iter_lines(paths: Union[PathLike, Iterable[PathLike]], encoding: str = "utf-8") -> Iterator[str]:
    """Yield non-empty, stripped lines from one or more files."""
    for path in _as_list(paths):
        with _open(path, encoding) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line


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
