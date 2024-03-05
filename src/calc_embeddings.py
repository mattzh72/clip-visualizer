from collections import defaultdict
from tqdm import tqdm
import pickle

from clip_utils import load_model, load_tokenizer, get_text_embed
from utils import read_nouns, generate_number_words_up_to_hundred


def main():
    model, tokenizer = load_model(), load_tokenizer()

    nouns = read_nouns()
    number_words = generate_number_words_up_to_hundred()
    number_buckets = defaultdict(list)
    noun_buckets = defaultdict(list)

    for number in tqdm(number_words, desc="Get embeddings"):
        for noun in nouns:
            sentence = f'{number} {noun}'
            if number != 'one':
                sentence += 's'
            
            embedding = get_text_embed(sentence, model, tokenizer)
            number_buckets[number].append(embedding)
            noun_buckets[noun].append(embedding)

    with open('../results/embedding_number_buckets.pkl', 'wb') as outfile:
        pickle.dump(number_buckets, outfile)

    with open('../results/embedding_noun_buckets.pkl', 'wb') as outfile:
        pickle.dump(noun_buckets, outfile)

    
if __name__ == "__main__":
    main()
