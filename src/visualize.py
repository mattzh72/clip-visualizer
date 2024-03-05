import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

from utils import generate_number_words_up_to_hundred, read_nouns


def set_titles_numbers(ax):
    ax.set_xlabel('Number Bucket (e.g. thirty-two)')
    ax.set_ylabel('Number Bucket (e.g. thirty-two)')
    ax.set_title("Average similarity scores across CLIP embeddings, bucketed by the number.")


def set_titles_nouns(ax):
    ax.set_xlabel('Noun Bucket (e.g. cat)')
    ax.set_ylabel('Noun Bucket (e.g. kangaroo)')
    ax.set_title("Average similarity scores across CLIP embeddings, bucketed by the noun.")


def main(bucket_by_number):
    bucket_type = 'number' if bucket_by_number else 'noun'

    if bucket_by_number:
        items = generate_number_words_up_to_hundred()
    else:
        items = read_nouns()

    with open(f'../results/similarity_{bucket_type}_buckets.pkl', 'rb') as infile:
        raw_similarity = pickle.load(infile)

    data = []
    for n1 in items:
        row = []
        for n2 in items:
            pair = (n1, n2)
            alternate = (n2, n1)

            if pair in raw_similarity:
                row.append(raw_similarity[pair])
            else:
                row.append(raw_similarity[alternate])

        data.append(row)

    data = np.array(data)
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    if bucket_by_number:
        set_titles_numbers(ax)
    else:
        set_titles_nouns(ax)

    plt.savefig(f'../results/{bucket_type}_bucket_visualization.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-by-number', action='store_true', help='Visualize based on number buckets.')
    args = parser.parse_args()
    print(args)

    main(args.bucket_by_number)