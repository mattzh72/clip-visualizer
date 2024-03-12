import configparser

from tqdm import tqdm
import pickle

from clip_utils import get_average_similarity
from utils import get_run_hash


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    base_emb_path = config['data']['embedding_base_path']
    base_sim_path = config['data']['sim_base_path']
    uuid = get_run_hash()
    buckets = config['global']['buckets'].split(',')
    
    for bucket in buckets:
        with open(f'{base_emb_path}/embedding_{bucket}_{uuid}.pkl', 'rb') as infile:
            embeddings = pickle.load(infile)

        embeddings = list(embeddings.items())
        pair_embeddings = {}

        for i in tqdm(range(len(embeddings)), desc='Calculate average similarity'):
            for j in range(i, len(embeddings)):
                (n1, e1), (n2, e2) = embeddings[i], embeddings[j]
                pair_embeddings[(n1, n2)] = get_average_similarity(e1, e2)

        with open(f'{base_sim_path}/sim_{bucket}_{uuid}.pkl', 'wb') as outfile:
            pickle.dump(pair_embeddings, outfile) 

if __name__ == "__main__":
    main()
