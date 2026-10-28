#!/usr/bin/env python3
"""Extract plain article text from a MediaWiki XML dump, one blank-line-separated document per article.

    curl -O https://dumps.wikimedia.org/amwiki/latest/amwiki-latest-pages-articles.xml.bz2
    python scripts/extract_wikipedia.py amwiki-latest-pages-articles.xml.bz2 amwiki.txt

Markup removal is approximate (templates, tables, references, file and category links, tags); it is
meant for word statistics, not for reading. Articles with fewer than --min-ethiopic Ethiopic characters
(redirects, lists, stubs made of templates) are skipped.
"""
import argparse
import bz2
import re
import xml.etree.ElementTree as ET

TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
TABLE = re.compile(r"\{\|.*?\|\}", re.S)
REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S)
TAG = re.compile(r"<[^>]+>")
FILE = re.compile(r"\[\[(?:File|Image|ፋይል|ስዕል|Category|መደብ):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
EXT = re.compile(r"\[https?://\S+\s*([^\]]*)\]")
TABLE_ROW = re.compile(r"^\s*[|!{}]")


def clean(text):
    text = REF.sub(" ", text)
    for _ in range(5):  # nested templates, innermost first
        stripped = TEMPLATE.sub(" ", text)
        if stripped == text:
            break
        text = stripped
    text = TABLE.sub(" ", text)
    text = FILE.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = EXT.sub(r"\1", text)
    text = TAG.sub(" ", text)
    text = re.sub(r"'{2,}|={2,}|^[*#:;]+", " ", text, flags=re.M)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line and not TABLE_ROW.match(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="pages-articles XML dump (.xml or .xml.bz2)")
    ap.add_argument("out", help="output text file")
    ap.add_argument("--min-ethiopic", type=int, default=50)
    a = ap.parse_args()
    opener = bz2.open if a.dump.endswith(".bz2") else open
    n = 0
    with opener(a.dump, "rb") as fh, open(a.out, "w", encoding="utf-8") as out:
        ns = text = None
        for _, el in ET.iterparse(fh):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "ns":
                ns = el.text
            elif tag == "text":
                text = el.text or ""
            elif tag == "page":
                if ns == "0":
                    body = clean(text)
                    if sum("ሀ" <= c <= "፿" for c in body) >= a.min_ethiopic:
                        out.write(body + "\n\n")
                        n += 1
                el.clear()
                ns = text = None
    print(f"wrote {n} articles to {a.out}")


if __name__ == "__main__":
    main()
