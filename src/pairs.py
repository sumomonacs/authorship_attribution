import random

# build a contrastive-learning pair list with author metadata.
def generate_contrastive_pairs(df, pos_per_author=1000, neg_pairs=5000, seed=42):

    random.seed(seed)

    # map author to list
    label_to_texts = {
        author: df[df.author == author].excerpt.tolist()
        for author in df.author.unique()
    }
    authors = list(label_to_texts)

    pairs = []

    #positive pairs (same author) 
    for author, texts in label_to_texts.items():
        # upper-bound by combinatorial maximum
        needed = min(pos_per_author, len(texts) * (len(texts) - 1) // 2)
        for _ in range(needed):
            a, b = random.sample(texts, 2)
            pairs.append((a, b, 1, author, author))

    # negative pairs (different authors)
    for _ in range(neg_pairs):
        a1, a2 = random.sample(authors, 2)     # uniform over author pairs
        pairs.append((
            random.choice(label_to_texts[a1]),
            random.choice(label_to_texts[a2]),
            0,
            a1,
            a2,
        ))

    random.shuffle(pairs)
    return pairs