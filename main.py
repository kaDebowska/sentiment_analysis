import re


def tokenize(text):
    return re.findall(r'\b[^\d\W]+\b', text.lower())


def stop_words(filename):
    stopwords = []
    with open(filename, encoding="utf-8") as infile:
        for line in infile:
            stopwords += line.strip().split(', ')
    return stopwords


def remove_stop_words(tokens):
    stopwords = stop_words('stopwords.txt')
    return [token for token in tokens if token not in stopwords]


def create_feature_dictionary(filename):
    feature_dict = {}
    with open(filename, encoding="utf-8") as infile:
        for line in infile:
            parts = line.split('__label__')
            text = parts[0].strip()
            label = '__label__' + parts[1].strip()

            tokens = remove_stop_words(tokenize(text))

            if label not in feature_dict:
                feature_dict[label] = {}

            for token in tokens:
                if token not in feature_dict[label]:
                    feature_dict[label][token] = 1
                else:
                    feature_dict[label][token] += 1

        return feature_dict


def sort_dictionary(feature_dict):
    for label, token_freq in feature_dict.items():
        feature_dict[label] = dict(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
    return feature_dict


feature_dictionary = sort_dictionary(create_feature_dictionary('all.text.train.txt'))

# print(stop_words('stopwords.txt'))

# print(feature_dictionary)

for label, word_freq in feature_dictionary.items():
    print(f"\nEtykieta: {label}")
    for word, freq in word_freq.items():
        print(f"{word}: {freq}")
