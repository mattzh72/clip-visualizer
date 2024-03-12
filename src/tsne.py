import pickle 
import argparse 

from tqdm import tqdm 

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main(bucket_by_number, use_cache):
    # Your embeddings dictionary
    # embeddings = {'label1': [embedding1, embedding2, ...], 'label2': [...], ...}
    bucket_type = 'number' if bucket_by_number else 'noun'
    embedding_cache = f'../../results/embedding_{bucket_type}_buckets.pkl'
    print(f"Opening {embedding_cache}")
    with open(embedding_cache, 'rb') as infile:
        embeddings = pickle.load(infile)

    # Flatten the embeddings and create labels
    all_embeddings = []
    labels = []
    for label, emb_list in tqdm(embeddings.items()):
        all_embeddings.extend(emb_list)  # Extend the list of all embeddings
        labels.extend([label] * len(emb_list))  # Extend the labels list with repeated labels

    if not use_cache:
        # Convert to numpy arrays for sklearn compatibility
        all_embeddings = np.array([emb.cpu().numpy() for emb in all_embeddings])
        labels = np.array(labels)

        tsne = TSNE(random_state=0, n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(all_embeddings)

        # Cache these results
        with open(f'../../results/tsne_{bucket_type}.pkl', 'wb') as outfile:
            pickle.dump(tsne_results, outfile) 
    else:
        with open(f'../../results/tsne_{bucket_type}.pkl', 'rb') as infile:
            tsne_results = pickle.load(infile)

    # Visualization
    # Convert labels to a numerical format
    label_to_id = {label: id for id, label in enumerate(set(labels))}
    numeric_labels = np.array([label_to_id[label] for label in labels])

    # Now use numeric_labels for the 'c' argument in plt.scatter
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=numeric_labels, cmap='tab10', alpha=0.6)

    # Create a legend with the original label names
    handles, _ = scatter.legend_elements()
    legend_labels = [k for k, v in sorted(label_to_id.items(), key=lambda item: item[1])]
    plt.legend(handles, legend_labels)
    plt.savefig(f'../../results/tsne_{bucket_type}.png', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-by-number', action='store_true', help='Visualize based on number buckets.')
    parser.add_argument('--use-cache', action='store_true', help='Use cached tsne result.')
    args = parser.parse_args()
    print(args)

    main(args.bucket_by_number, args.use_cache)