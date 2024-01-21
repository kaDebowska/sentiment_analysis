import re
from collections import defaultdict
import math


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


def generate_text_and_labels(filename):
    with open(filename, encoding="utf-8") as infile:
        for line in infile:
            parts = line.split('__label__')
            text = parts[0].strip()
            label = '__label__' + parts[1].strip()
            yield text, label


def create_feature_dictionary(generator):
    feature_dict = {}
    for text, label in generator:
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


def train_naive_bayes(feature_dict, labels):
    model = {}
    total_docs = sum(len(feature_dict[label]) for label in labels)

    for label in labels:
        model[label] = {'prior': len(feature_dict[label]) / total_docs, 'tokens': defaultdict(int)}

        for token, occurrence in feature_dict[label].items():
            model[label]['tokens'][token] = occurrence / sum(feature_dict[label].values())

    return model


def classify_text(model, text, labels):
    tokens = remove_stop_words(tokenize(text))
    scores = {}

    for label in labels:
        scores[label] = math.log(model[label]['prior'])

        for token in tokens:
            scores[label] += math.log(model[label]['tokens'].get(token, 1e-10))

    return max(scores, key=scores.get)


def evaluate_model(model, generator, labels):
    correct_predictions = 0
    total_samples = 0

    for text, true_label in generator:
        predicted_label = classify_text(model, text, labels)
        total_samples += 1

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy


def main():
    feature_dictionary = create_feature_dictionary(generate_text_and_labels('all.text.train.txt'))
    print("feature_dictionary")
    labels = feature_dictionary.keys()
    print("labels")

    labels_description = {
        '__label__meta_minus_m': 'Wypowiedź negatywna',
        '__label__meta_plus_m': 'Wypowiedź pozytywna',
        '__label__meta_zero': 'Wypowiedź neutralna',
        '__label__meta_amb': 'Wypowiedź ambiwalentan'
    }

    naive_bayes_model = train_naive_bayes(feature_dictionary, labels)
    print("naive_bayes_model")
    sample_texts = [
        "Obsługa była miła i profesjonalna",
        "Hotel był wygodny, choć mały",
        "Najgorszy film, jaki widziałem w tym roku.",
        "Byłem na wizycie w poniedziałek",
        "Szczerze polecam, Pani doktor bardzo miła i zaangażowana. ",
        "ta wizyta nie dała mi nic poza frustracją/dodatkowymi nerwami ",
        "Jakość wykonania pozostawia wiele do życzenia. Odradzam",
        "Urządzenie działa poprawnie, choć z taką cenę spodziewałem się czegoś więcej.",
        "Treść lekka aczkolwiek przeciętna .",
        "Niestety akumulator w tej szczoteczce jest słaby ( technologia NiMH ) .",
        "Przyznać trzeba że lodówka bardzo funkcjonalna i dobrze zaprojektowana z punktu użytkowego , "
        "ale jeśli chodzi o wytrzymałość to im nie wyszło .",
        "Pokoje są dramatyczne po odsunięciu łóżka okazało się że jest tam jakiś żuk. Brudno śmierdzi. Omijać szerokim łukiem."
    ]

    test_data_generator = generate_text_and_labels('all.text.test.txt')
    print("test_data_generator")

    accuracy = evaluate_model(naive_bayes_model, test_data_generator, labels)
    print("accuracy")

    print(f"Dokładność modelu na zbiorze testowym: {accuracy * 100:.2f}%")

    for sample_text in sample_texts:
        classified_label = classify_text(naive_bayes_model, sample_text, labels)
        print(f"\nKlasyfikacja dla tekstu: {sample_text}\nPrzewidziana etykieta: {labels_description[classified_label]}")
    print("classified_label")

if __name__ == "__main__":
    main()