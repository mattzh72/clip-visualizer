from tqdm import tqdm
import pickle

from clip_utils import get_average_similarity


def main():
    with open('../results/embeddings_checkpoint.pkl', 'rb') as infile:
        embeddings = pickle.load(infile)

    embeddings = list(embeddings.items())
    pair_embeddings = {}

    for i in tqdm(range(len(embeddings)), desc='Calculate average similarity'):
        for j in range(i, len(embeddings)):
            (n1, e1), (n2, e2) = embeddings[i], embeddings[j]
            pair_embeddings[(n1, n2)] = get_average_similarity(e1, e2)

    with open('../results/similarity.pkl', 'wb') as outfile:
        pickle.dump(pair_embeddings, outfile) 

    
if __name__ == "__main__":
    main()
