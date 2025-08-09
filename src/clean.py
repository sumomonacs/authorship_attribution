import re

# preprocess the book
def clean_book(file_path, opening_line):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # remove Gutenberg header/footer
    start = re.search(r"\*\*\* *START OF.*?\*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\* *END OF.*?\*\*\*", text, re.IGNORECASE)
    if start and end:
        text = text[start.end():end.start()]
    elif start:
        text = text[start.end():]
    elif end:
        text = text[:end.start()]

    # trim before true narrative opening
    idx = text.find(opening_line)
    if idx != -1:
        text = text[idx:]
    else:
        print(f"Opening line not found in: {file_path}")

    # remove multiline [Illustration: ... ] or [ ... ] blocks
    text = re.sub(r"\[\s*Illustration:.*?\]+", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\s*_?Copyright.*?\]+", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\s*.*?\s*\]", "", text, flags=re.DOTALL)  # generic fallback

    return text.strip()
