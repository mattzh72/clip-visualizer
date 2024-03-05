def read_nouns():
    with open('../data/animals.txt', 'r') as file:
        words = file.readlines()
    return [word.strip().lower() for word in words]


def generate_number_words_up_to_hundred(limit_exclusive=51):
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    
    number_words = []
    
    for i in range(1, limit_exclusive):
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