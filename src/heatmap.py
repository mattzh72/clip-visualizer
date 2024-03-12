import pickle
import configparser

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

from utils import generate_number_words_up_to_hundred, read_nouns, get_run_hash, read_verbs


def set_titles(ax, bucket):
    ax.set_xlabel(f'{bucket} Bucket')
    ax.set_ylabel(f'{bucket} Bucket')
    ax.set_title(f"Average similarity scores across CLIP embeddings, bucketed by the {bucket}.")

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    base_sim_path = config['data']['sim_base_path']
    bucket_type = config['heatmap']['bucket']
    output = config['heatmap']['output']
    with_ticks = config['heatmap']['with_ticks']
    uuid = get_run_hash()

    if bucket_type == 'number':
        items = generate_number_words_up_to_hundred()
    elif bucket_type == 'verb':
        items = read_verbs()
    else:
        items = read_nouns()

    with open(f'{base_sim_path}/sim_{bucket_type}_{uuid}.pkl', 'rb') as infile:
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
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(data)
    ax.xaxis.set_ticks_position('top')

    ax.set_xticks(np.arange(len(items))) # Correct
    ax.set_yticks(np.arange(len(items))) # Correct


    # Example tick labels. Replace with your actual labels if needed.
    xtick_labels = items # Customize these as per your requirement
    ytick_labels = items # Customize these as per your requirement
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)

    # Rotate the tick labels for better visibility if necessary
    if with_ticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    set_titles(ax, bucket_type.capitalize())

    plt.savefig(f'{output}/{bucket_type}_{uuid}.png')

if __name__ == "__main__":
    main()