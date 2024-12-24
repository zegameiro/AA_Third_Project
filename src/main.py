from collections import Counter
from nltk.corpus import stopwords

import nltk
import re
import random
import math
import heapq
import timeit

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

def preprocess_text(file_path):
    """Preprocess text by removing headers, footers, punctuation, stopwords and converting to lowercase"""

    with open(file_path, 'r') as file:
        text = file.read()

    # Remove Project Gutenberg headers and footers
    text = re.sub(r"(\*\*\* START OF THIS PROJECT GUTENBERG.*?\*\*\*|\*\*\* END OF THIS PROJECT GUTENBERG.*)", "", text, flags=re.DOTALL)

    # Remove punctuation and convert to lowercase
    text = re.sub(r"[^\w\s]", "", text).lower()

    # Tokenize and remove stopwords
    words = text.split()
    words = [token for token in words if token not in stopwords]

    return words

def exact_word_count(words) -> tuple[dict, float]:
    """Count the exact frequency of each word in the text"""

    start = timeit.timeit()

    exact_words = Counter(words)

    end = timeit.timeit()

    return exact_words, end - start

def decreasing_probability_counter(words, n) -> tuple[dict, float, int]:
    """Estimate word frequencies using decreasing probability counter (1 / sqrt(2)^k)"""

    start = timeit.timeit()

    counts = {}

    for word in words:

        if word not in counts:
            counts[word] = 1
        else:
            # Update count with decreasing probability
            k = counts[word]
            if random.random() < 1 / math.sqrt(2) ** k:
                counts[word] += 1

    end = timeit.timeit()

    # Get the top n words
    top_n_words = {k: v for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]}

    return top_n_words, end - start

def space_saving_counter(words, n) -> tuple[dict, float, int]:
    """Space saving Algorithm to track top n frequent words"""

    start = timeit.timeit()

    frequency_map = {}
    heap = []

    for word in words:

        if word in frequency_map:
            # Increment the frequency if the word is already tracked
            frequency_map[word] += 1
            heapq.heapify(heap)

        elif len(frequency_map) < n:
            # Track the word if there is space
            frequency_map[word] = 1
            heapq.heappush(heap, (1, word))

        else:
            # Replace the least frequent word
            min_frequency, min_word = heapq.heappop(heap)
            del frequency_map[min_word]
            frequency_map[word] = min_frequency + 1
            heapq.heappush(heap, (min_frequency + 1, word))

    end = timeit.timeit()

    return sorted(frequency_map.items(), key=lambda x: x[1], reverse=True), end - start

def main():

    words = preprocess_text("../data/indian_queen.txt")
    n = 10

    exact_words, exact_time = exact_word_count(words)
    print(f"Top 10 words from exact counter: {exact_words.most_common(n)}\nTime taken: {exact_time}\n")

    approx_words, approx_time = decreasing_probability_counter(words, n)
    print(f"Approximate word count: {approx_words}\nTime taken: {approx_time}\n")

    top_n_words, space_saving_time = space_saving_counter(words, n)
    print(f"Top 10 words: {top_n_words}\nTime taken: {space_saving_time}\n")

if __name__ == "__main__":
    main()




    