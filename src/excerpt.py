import re
import random
import string

# get the excerpts inside the paragraph, and roughly uniform across the book
def split_into_excerpts(text, min_words=200, max_words=400, stride=100,
                        buffer_from_end=500, max_excerpts=None, seed=42):
    random.seed(seed)

    # split into paragraphs and drop empties
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # work out how many words total to skip the tail
    total_words = len(text.split())

    # keep only paragraphs that lie before the tail zone
    running_total = 0
    usable_paragraphs = []
    for p in paragraphs:
        running_total += len(p.split())
        if running_total > total_words - buffer_from_end:
            break
        usable_paragraphs.append(p)

    excerpts = []

    # sample within each usable paragraph
    for para in usable_paragraphs:
        words = para.split()
        if len(words) < min_words:
            continue

        starts = list(range(0, len(words) - min_words, stride))
        random.shuffle(starts)           # randomise start positions

        for start in starts:
            chunk = words[start : start + max_words]
            if len(chunk) >= min_words:
                excerpts.append(" ".join(chunk))
                break                    # take at most one per paragraph

    random.shuffle(excerpts)             # shuffle global order

    if max_excerpts is not None:
        excerpts = excerpts[:max_excerpts]

    return excerpts


# a helper function to define the heading form
_heading_re = re.compile(
    r"""^              # start
        (?:chapter|book|section)?  # optional label
        \s*                    # whitespace
        [ivxlcdm]+             # roman numerals
        [\.\s]*$               # optional dot/space, end
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# determine standalone heading such as 'III' or 'CHAPTER XL'.
def _looks_like_heading(para):
    # strip punctuation & collapse spaces for robust matching
    clean = para.translate(str.maketrans("", "", string.punctuation)).strip()
    if _heading_re.match(clean):
        return True
    # ultra-short indicates a heading
    return len(clean.split()) <= 3


    
# concatenate up to max_para non-heading paragraphs to build excerpts, skip the final words to avoid spoilers
def multi_para_excerpts(text, min_words=200, max_words=400, max_paras=4,
    para_stride=1, buffer_from_end=20, max_excerpts=None, seed=42):
    random.seed(seed)

    # split paras & filter headings
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paras = [p for p in raw_paras if not _looks_like_heading(p)]

    total_words = len(text.split())
    spoiler_cutoff = total_words - buffer_from_end

    excerpts = []
    i, words_seen = 0, 0

    while i < len(paras) and words_seen < spoiler_cutoff:
        chunk, chunk_words, added = [], 0, 0
        for j in range(i, min(i + max_paras, len(paras))):
            p_words = paras[j].split()
            if words_seen + chunk_words + len(p_words) > spoiler_cutoff:
                break
            chunk.append(paras[j])
            chunk_words += len(p_words)
            added += 1
            if chunk_words >= min_words:
                break

        if min_words <= chunk_words <= max_words:
            excerpts.append(" ".join(chunk))

        # advance
        i += max(added, para_stride)
        words_seen += sum(len(p.split()) for p in paras[i - added : i])

        if max_excerpts and len(excerpts) >= max_excerpts:
            break

    random.shuffle(excerpts)
    return excerpts