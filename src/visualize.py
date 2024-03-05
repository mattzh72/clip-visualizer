import pickle

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

from utils import generate_number_words_up_to_hundred


if __name__ == "__main__":
    numbers = generate_number_words_up_to_hundred()

    with open('../results/similarity.pkl', 'rb') as infile:
        raw_similarity = pickle.load(infile)

    data = []
    for n1 in numbers:
        row = []
        for n2 in numbers:
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

    # # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(numbers)), labels=numbers)
    # ax.set_yticks(np.arange(len(numbers)), labels=numbers)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(numbers)):
    #     for j in range(len(numbers)):
    #         text = ax.text(j, i, data[i, j],
    #                     ha="center", va="center", color="w")

    ax.set_title("Average similarity scores for different number buckets.")
    plt.savefig('../results/visualization.png')