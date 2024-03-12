import configparser

from collections import defaultdict
from tqdm import tqdm
import pickle

from clip_utils import load_model, get_text_embed
from utils import read_nouns, read_verbs, generate_number_words_up_to_hundred, get_run_hash


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    base_output_path = config['data']['embedding_base_path']
    model = load_model()

    nouns = read_nouns()
    verbs = read_verbs()
    number_words = generate_number_words_up_to_hundred()
    number_buckets = defaultdict(list)
    noun_buckets = defaultdict(list)
    verb_buckets = defaultdict(list)

    for number in tqdm(number_words, desc="Get embeddings"):
        for noun in nouns:
            for verb in verbs:
                sentence = f'{number} {noun}'
                if number != 'one':
                    sentence += 's'
                sentence += f' {verb}'

                embedding = get_text_embed(sentence, model)
                number_buckets[number].append(embedding)
                noun_buckets[noun].append(embedding)
                verb_buckets[verb].append(embedding)

    uuid = get_run_hash()
    with open(f'{base_output_path}/embedding_number_{uuid}.pkl', 'wb') as outfile:
        pickle.dump(number_buckets, outfile)

    with open(f'{base_output_path}/embedding_noun_{uuid}.pkl', 'wb') as outfile:
        pickle.dump(noun_buckets, outfile)

    with open(f'{base_output_path}/embedding_verb_{uuid}.pkl', 'wb') as outfile:
        pickle.dump(verb_buckets, outfile)

    
if __name__ == "__main__":
    main()
