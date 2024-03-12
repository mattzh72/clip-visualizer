import configparser
import shortuuid

config = configparser.ConfigParser()
config.read('config.ini')

def get_run_hash():
    noun_file = config['data']['noun_file']
    verb_file = config['data']['verb_file']
    count_limit = config['data']['count_limit']

    return short_hash_shortuuid(noun_file + verb_file + count_limit)

def short_hash_shortuuid(text):
    # `shortuuid` can generate a short uuid based on SHA-1 hash
    shortuuid.set_alphabet("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return shortuuid.uuid(name=text)

def read_nouns():
    file = config['data']['noun_file']
    
    with open(file, 'r') as file:
        words = file.readlines()
    return [word.strip().lower() for word in words]

def read_verbs():
    file = config['data']['verb_file']
    
    with open(file, 'r') as file:
        words = file.readlines()
    return [word.strip().lower() for word in words]

def generate_number_words_up_to_hundred():
    count_limit = int(config['data']['count_limit'])

    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    number_words = []
    
    for i in range(1, count_limit):
        if i < 10:
            number_words.append(units[i])
        elif 10 <= i < 20:
            number_words.append(teens[i-10])
        else:
            ten, unit = divmod(i, 10)
            if unit == 0:
                number_words.append(tens[ten])
            else:
                number_words.append(tens[ten] + "-" + units[unit])
        
    return number_words