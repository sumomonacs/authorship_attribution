'''
    This file extracts excerpts from the books.
'''
import os
import sys
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.excerpt import split_into_excerpts, multi_para_excerpts
from src.config import CLEANED_DIR, EXCERPT_FILE

AUTHOR_MAP = {
    "bronte":  "Charlotte Brontë",
    "eliot":   "George Eliot",
    "james":   "Henry James",
    "woolf":   "Virginia Woolf",
    "wharton": "Edith Wharton",
}

# different criteria for shorter books(can extract across paragraphs)
BOOK_OVERRIDES = {
    "woolf_mrs_dalloway": {
        "extractor":  "multi_para",
        "min_words":  120,
        "max_paras":  6,
        "para_stride": 1,
    },
    "wharton_age_of_innocence": {
        "extractor":  "multi_para",
        "min_words":  180,
        "max_paras":  4,
        "para_stride": 1,
    },
    "james_turn_of_the_screw": {
        "extractor":  "multi_para",
        "min_words":  180,
        "max_paras":  5,
        "para_stride": 1,
    },
}

# get author name from the file name
def author_from_fname(fname):
    for key, name in AUTHOR_MAP.items():
        if fname.startswith(key):
            return name
    return "Unknown"

# build the excerpt dataset
def build_excerpt_dataset( min_words=200, max_words=400, max_excerpts_per_book=120,
    max_excerpts_per_author=200, stride= 300,):
    # collect files per author
    author_books = defaultdict(list)
    for fname in os.listdir(CLEANED_DIR):
        if fname.endswith(".txt"):
            author_books[author_from_fname(fname)].append(fname)

    rows = []

    # iterate author to book
    for author, books in author_books.items():
        per_book_cap = min(
            max_excerpts_per_book,
            max_excerpts_per_author // max(len(books), 1)
        )

        for fname in books:
            base = os.path.splitext(fname)[0]         
            path = os.path.join(CLEANED_DIR, fname)
            with open(path, encoding="utf-8") as f:
                text = f.read()

            override = BOOK_OVERRIDES.get(base)

            if override and override["extractor"] == "multi_para":
                excerpts = multi_para_excerpts(
                    text,
                    min_words   = override.get("min_words",   180),
                    max_words   = max_words,
                    max_paras   = override.get("max_paras",   4),
                    para_stride = override.get("para_stride", 1),
                    max_excerpts= per_book_cap,
                )
            else:
                excerpts = split_into_excerpts(
                    text,
                    min_words   = min_words,
                    max_words   = max_words,
                    stride      = stride,
                    max_excerpts= per_book_cap,
                )

            # append rows no matter which extractor was used
            for ex in excerpts:
                rows.append({"excerpt": ex, "author": author, "book": base})

    df = pd.DataFrame(rows)

    # sanity reports
    print("\nExcerpt distribution (by author)")
    print(df.groupby("author").size().to_string())

    print("\nExcerpt distribution (by book)")
    print(df.groupby(["author", "book"]).size().to_string())

    print("---------------------------------------")
    print(f"Total excerpts: {len(df)}\n")

    return df

def main():
    df = build_excerpt_dataset()
    df.to_csv(EXCERPT_FILE, index=False)
    print(f"Saved to {EXCERPT_FILE}")

if __name__ == "__main__":
    main()
