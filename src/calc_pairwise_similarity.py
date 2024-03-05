from tqdm import tqdm
import pickle
import argparse

from clip_utils import get_average_similarity


def main(bucket_by_number):
    bucket_type = 'number' if bucket_by_number else 'noun'

    with open(f'../results/embedding_{bucket_type}_buckets.pkl', 'rb') as infile:
        embeddings = pickle.load(infile)

    embeddings = list(embeddings.items())
    pair_embeddings = {}

    for i in tqdm(range(len(embeddings)), desc='Calculate average similarity'):
        for j in range(i, len(embeddings)):
            (n1, e1), (n2, e2) = embeddings[i], embeddings[j]
            pair_embeddings[(n1, n2)] = get_average_similarity(e1, e2)

    with open(f'../results/similarity_{bucket_type}_buckets.pkl', 'wb') as outfile:
        pickle.dump(pair_embeddings, outfile) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-by-number', action='store_true', help='Visualize based on number buckets.')
    args = parser.parse_args()
    print(args)

    main(args.bucket_by_number)
