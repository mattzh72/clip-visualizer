from collections import defaultdict
from tqdm import tqdm
import pickle

from clip_utils import load_model, load_tokenizer, get_text_embed
from utils import read_nouns, generate_number_words_up_to_hundred


def main():
    model, tokenizer = load_model(), load_tokenizer()

    nouns = read_nouns()
    number_words = generate_number_words_up_to_hundred()
    embeddings = defaultdict(list)

    for number in tqdm(number_words, desc="Get embeddings"):
        for n in nouns:
            sentence = f'{number} {n}'
            if number != 'one':
                sentence += 's'
            embeddings[number].append(get_text_embed(sentence, model, tokenizer))

    with open('../results/embeddings_checkpoint.pkl', 'wb') as outfile:
        pickle.dump(embeddings, outfile)

    
if __name__ == "__main__":
    main()
