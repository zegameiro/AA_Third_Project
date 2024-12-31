from nltk.corpus import stopwords

import nltk
import re
import random
import math
import matplotlib.pyplot as plt
import os
import argparse


DATA_STREAM_REPETITIONS = 10
K_VALUES = [20, 35, 100]

def preprocess_text(file_path, stopwords):
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

def exact_word_count(words):
    """Count the exact frequency of each word in the text"""

    counter = {}

    for word in words:
        if word in counter:
            counter[word] += 1
        else:
            counter[word] = 1

    return counter

def decreasing_probability_counter(exact_counts, words):
    """Estimate word frequencies using decreasing probability counter (1 / sqrt(2)^k)"""

    errors = []
    best_counter = None

    for _ in range(DATA_STREAM_REPETITIONS):
        counts = {}

        for word in words:  
            count = 0

            if word in counts:
                count = counts[word]

            p = random.uniform(0, 1)

            if p < 1 / (math.sqrt(2) ** count):
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1

        error = calculate_error(exact_counts=exact_counts, method_count=counts)
        if not best_counter or error < min(errors):
            best_counter = counts

        errors.append(error)

    average_error = sum(errors) / DATA_STREAM_REPETITIONS
    highest_error = max(errors)
    lowest_error = min(errors)

    print("\nDecreasing Probability Counter errors:\n")
    print(f"Average error: {average_error}\n")
    print(f"Highest error: {highest_error}\n")
    print(f"Lowest error: {lowest_error}\n")

    return best_counter


def space_saving_counter(words, k):
    """Space-saving algorithm to approximate top-k frequent words."""
    
    counter = {}

    for word in words:

        if word in counter:
            counter[word] += 1

        elif len(counter) < k:
            counter[word] = 1
        else:
            min_word, _ = min(counter.items(), key=lambda x: x[1])
            smallest_value = counter[min_word]
            del counter[min_word]
            counter[word] = smallest_value + 1

    return counter


def calculate_error(exact_counts, method_count):
    """Calculate the error between the exact count and the approximate count"""

    error = sum(abs(method_count[word] - exact_counts[word]) for word in method_count)
    return error


def compute_errors(exact_counts, approximate_counts, filename):

    absolute_errors = {}
    relative_errors = {}

    for word in approximate_counts.keys():
        absolute_error = abs(approximate_counts[word] - exact_counts[word])
        relative_error = absolute_error / exact_counts[word]
        absolute_errors[word] = absolute_error
        relative_errors[word] = relative_error

    min_absolute_error, avg_absolute_error, max_absolute_error = min(absolute_errors.values()), sum(absolute_errors.values()) / len(absolute_errors), max(absolute_errors.values())
    min_relative_error, avg_relative_error, max_relative_error = min(relative_errors.values()), sum(relative_errors.values()) / len(relative_errors), max(relative_errors.values())

    top_10_words = sorted(approximate_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    create_chart(name=f"{filename}_absolute_errors", counts=absolute_errors, top_10_words=top_10_words)
    create_chart(name=f"{filename}_relative_errors", counts=relative_errors, top_10_words=top_10_words)

    return min_absolute_error, avg_absolute_error, max_absolute_error, min_relative_error, avg_relative_error, max_relative_error


def create_chart(name, counts, top_10_words):

    errors = []
    words = []
    for word, _ in top_10_words:
        words.append(word)
        errors.append(counts[word])

    plt.bar(words, errors)
    plt.ylabel('Error')
    plt.xlabel('Words')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"../images/{name}.png")
    plt.close()


def main():

    arg_parser = argparse.ArgumentParser(description="Approximate word count algorithms")
    arg_parser.add_argument("--file", type=str, help="Path to the text file to process", required=True)

    args = arg_parser.parse_args()
    
    try:
        nltk.data.find('stopwords')
    except LookupError:
        nltk.download('stopwords')

    language = ""

    if "en" in args.file:
        language = "english"
    elif "sp" in args.file:
        language = "spanish"
    elif "ger" in args.file:
        language = "german"
    elif "hun" in args.file:
        language = "hungarian"
    else:
        raise ValueError("Language not supported")

    language_stopwords = set(stopwords.words(language))
    
    if not os.path.exists(args.file):
        raise FileNotFoundError("File not found")

    words = preprocess_text(file_path=args.file, stopwords=language_stopwords)

    os.makedirs("../images/", exist_ok=True)

    exact_counter = exact_word_count(words)

    decreasing_probability_counter_results = decreasing_probability_counter(exact_counts=exact_counter, words=words)
    min_absolute_error_appr, avg_absolute_error_appr, max_absolute_error_appr, min_relative_error_appr, avg_relative_error_appr, max_relative_error_appr = compute_errors(exact_counts=exact_counter, approximate_counts=decreasing_probability_counter_results, filename=f"don_quixote_{language}_approximate_counter")

    space_saving_counter_results = {}
    average_errors = {}
    for k in K_VALUES:
        space_saving_counter_results[k] = space_saving_counter(words, k)
        min_absolute_error, avg_absolute_error, max_absolute_error, min_relative_error, avg_relative_error, max_relative_error = compute_errors(exact_counts=exact_counter, approximate_counts=space_saving_counter_results[k], filename=f"don_quixote_{language}_k_{k}_space_saving_counter")
        average_errors[k] = {
            "min_absolute_error": min_absolute_error,
            "avg_absolute_error": avg_absolute_error,
            "max_absolute_error": max_absolute_error,
            "min_relative_error": min_relative_error,
            "avg_relative_error": avg_relative_error,
            "max_relative_error": max_relative_error
        }

    os.makedirs("../results/", exist_ok=True)

    with open(f"../results/don_quixote_{language}_results.txt", "w") as file:
        file.write(f"Don Quixote {language} version \n\n")

        top_10_exact = sorted(exact_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        file.write(f"Exact Counter:\n")
        for word, count in top_10_exact:
            file.write(f"{word}: {count}\n")

        file.write("\n")

        top_10_decreasing = sorted(decreasing_probability_counter_results.items(), key=lambda x: x[1], reverse=True)[:10]
        file.write(f"Decreasing Probability Counter:\n")
        file.write(f"   Top 10 most Frequent Words:\n")
        for word, count in top_10_decreasing:
            file.write(f"       {word}: {count}\n")
        file.write(f"   Errors:\n")
        file.write(f"       Min Absolute Error: {min_absolute_error_appr}\n")
        file.write(f"       Average Absolute Error: {avg_absolute_error_appr}\n")
        file.write(f"       Max Absolute Error: {max_absolute_error_appr}\n")
        file.write(f"       Min Relative Error: {min_relative_error_appr}\n")
        file.write(f"       Average Relative Error: {avg_relative_error_appr}\n")
        file.write(f"       Max Relative Error: {max_relative_error_appr}\n")
        
        file.write("\n")
        
        for k in K_VALUES:
            file.write(f"Space Saving Counter with k = {k}:\n")
            file.write(f"   Top 10 most Frequent Words:\n")
            top_10_space_saving = sorted(space_saving_counter_results[k].items(), key=lambda x: x[1], reverse=True)[:10]
            for word, count in top_10_space_saving:
                file.write(f"       {word}: {count}\n")
            file.write(f"   Errors:\n")
            file.write(f"       Min Absolute Error: {average_errors[k]['min_absolute_error']}\n")
            file.write(f"       Average Absolute Error: {average_errors[k]['avg_absolute_error']}\n")
            file.write(f"       Max Absolute Error: {average_errors[k]['max_absolute_error']}\n")
            file.write(f"       Min Relative Error: {average_errors[k]['min_relative_error']}\n")
            file.write(f"       Average Relative Error: {average_errors[k]['avg_relative_error']}\n")
            file.write(f"       Max Relative Error: {average_errors[k]['max_relative_error']}\n")
            file.write("\n")

    print("Results saved to results folder")
    print("Charts saved to images folder")

if __name__ == "__main__":
    main()




    