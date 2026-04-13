import os
import re
from collections import defaultdict

print("Initializing Search Engine...")

# Stop words
stop_words = {"a", "an", "the", "they", "these", "this", "for", "is", "are", "was", "of", "or", "and", "does", "will", "whose"}

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[{}[\]<>=().,;\'"?#!\-:]', ' ', text)
    words = text.split()
    return [w for w in words if w not in stop_words]


# Inverted index
inverted_index = defaultdict(list)

# Path
webpages_path = "Datasets/Assignment 4- datasets/Assignment 4- datasets/Q2- webSearch/webpages"

# Build index
for page in os.listdir(webpages_path):
    with open(os.path.join(webpages_path, page), "r", errors="ignore") as f:
        words = clean_text(f.read())

        for pos, word in enumerate(words):
            inverted_index[word].append((page, pos))


print("Total words indexed:", len(inverted_index))

print("Step Q2-1 successful ✅")

print("\n===== QUERY FUNCTIONS =====")


# Function 1: Find pages containing word
def queryFindPagesWhichContainWord(word):
    word = word.lower()

    if word not in inverted_index:
        print(f"No webpage contains word {word}")
        return

    pages = set([entry[0] for entry in inverted_index[word]])
    print(", ".join(pages))


# Function 2: Find positions of word in a page
def queryFindPositionsOfWordInAPage(word, page):
    word = word.lower()

    if word not in inverted_index:
        print(f"Webpage {page} does not contain word {word}")
        return

    positions = [pos for (p, pos) in inverted_index[word] if p == page]

    if not positions:
        print(f"Webpage {page} does not contain word {word}")
    else:
        print(", ".join(map(str, positions)))

print("\n===== TESTING QUERIES =====")

queryFindPagesWhichContainWord("stack")
queryFindPagesWhichContainWord("delhi")

queryFindPositionsOfWordInAPage("stack", "stack_datastructure_wiki")
queryFindPositionsOfWordInAPage("data", "stack_oracle")

print("\n===== PROCESSING actions.txt =====")

actions_path = "Datasets/Assignment 4- datasets/Assignment 4- datasets/Q2- webSearch/actions.txt"

with open(actions_path, "r") as f:
    for line in f:
        parts = line.strip().split()

        if not parts:
            continue

        if parts[0] == "queryFindPagesWhichContainWord":
            word = parts[1]
            print(f"\nQuery: Pages containing '{word}'")
            queryFindPagesWhichContainWord(word)

        elif parts[0] == "queryFindPositionsOfWordInAPage":
            word = parts[1]
            page = parts[2]
            print(f"\nQuery: Positions of '{word}' in '{page}'")
            queryFindPositionsOfWordInAPage(word, page)