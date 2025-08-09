'''
    This file downloads the books and preprocess them.
'''
import os
import requests
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.clean import clean_book
from src.config import RAW_DIR, CLEANED_DIR

# get the raw text files from project Gutenberg
books = {
    # Charlotte Brontë
    "bronte_jane_eyre.txt": "https://www.gutenberg.org/files/1260/1260-0.txt",
    "bronte_villette.txt": "https://www.gutenberg.org/files/9182/9182-0.txt",

    # George Eliot
    "eliot_mill_on_the_floss.txt": "https://www.gutenberg.org/files/6688/6688-0.txt",
    "eliot_middlemarch.txt": "https://www.gutenberg.org/files/145/145-0.txt",

    # Henry James
    "james_turn_of_the_screw.txt": "https://www.gutenberg.org/files/209/209-0.txt",
    "james_portrait_of_a_lady.txt": "https://www.gutenberg.org/files/2833/2833-0.txt",

    # Virginia Woolf
    "woolf_mrs_dalloway.txt": "https://www.gutenberg.org/files/63107/63107-0.txt",
    "woolf_to_the_lighthouse.txt": "https://www.gutenberg.org/files/144/144-0.txt",  
    "woolf_orlando.txt" : "https://gutenberg.net.au/ebooks02/0200331.txt",

    # Edith Wharton
    "wharton_age_of_innocence.txt": "https://www.gutenberg.org/files/421/421-0.txt",
    "wharton_house_of_mirth.txt": "https://www.gutenberg.org/files/284/284-0.txt"
}


# openings of the books
OPENINGS = {
    # Charlotte Brontë
    "bronte_jane_eyre.txt": "There was no possibility of taking a walk that day",
    "bronte_villette.txt": "My godmother lived in a handsome house in the clean and ancient town ",

    # George Eliot
    "eliot_mill_on_the_floss.txt": "A wide plain, where the broadening Floss hurries on",
    "eliot_middlemarch.txt": "Miss Brooke had that kind of beauty which seems to be thrown",

    # Henry James
    "james_turn_of_the_screw.txt": "The story had held us, round the fire, sufficiently breathless,",
    "james_portrait_of_a_lady.txt": "Under certain circumstances there are few hours in life more agreeable",

    # Virginia Woolf
    "woolf_mrs_dalloway.txt": "Mrs Dalloway said she would buy the gloves herself.",
    "woolf_to_the_lighthouse.txt": "As the streets that lead from the Strand to the Embankment are very",
    "woolf_orlando.txt" : "He--for there could be no doubt of his sex,",
    
    # Edith Wharton
    "wharton_age_of_innocence.txt": "I will begin the story of my adventures with a certain morning early",
    "wharton_house_of_mirth.txt": "Selden paused in surprise. In the afternoon rush of the Grand"
}

# dowmload the books to the directory data
def download_books():
    for fname, url in books.items():
        print(f"Downloading {fname} ...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(os.path.join(RAW_DIR, fname), "w", encoding="utf-8") as f:
                f.write(r.text)
        else:
            print(f"Failed to download {fname}")
    print("All books downloaded into /data")

# get the cleaned version of the books
def get_cleaned():
    for fname, opening in OPENINGS.items():
        in_path = os.path.join(RAW_DIR, fname)
        out_path = os.path.join(CLEANED_DIR, fname)
        if os.path.exists(in_path):
            cleaned = clean_book(in_path, opening)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Cleaned: {fname}")
        else:
            print(f"File not found: {in_path}")

def main():
    download_books()
    get_cleaned()

if __name__ == "__main__":
    main()