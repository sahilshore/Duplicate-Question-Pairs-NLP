import re
import pickle
import numpy as np
import distance
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz


# ==========================
# Load necessary resources
# ==========================
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
STOP_WORDS = pickle.load(open('stopwords.pkl', 'rb'))


# ==========================
# Basic Word Features
# ==========================
def test_common_words(q1, q2):
    w1 = set(map(str.lower, q1.split()))
    w2 = set(map(str.lower, q2.split()))
    return len(w1 & w2)


def test_total_words(q1, q2):
    w1 = set(map(str.lower, q1.split()))
    w2 = set(map(str.lower, q2.split()))
    return len(w1) + len(w2)


# ==========================
# Token-based Features
# ==========================
def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8

    q1_tokens, q2_tokens = q1.split(), q2.split()
    if not q1_tokens or not q2_tokens:
        return token_features

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])

    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])  # last word same
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])    # first word same

    return token_features


# ==========================
# Length-based Features
# ==========================
def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3
    q1_tokens, q2_tokens = q1.split(), q2.split()

    if not q1_tokens or not q2_tokens:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features


# ==========================
# Fuzzy Matching Features
# ==========================
def test_fetch_fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]


# ==========================
# Text Preprocessing
# ==========================
def preprocess(q):
    q = str(q).lower().strip()

    # Replace special characters
    q = (q.replace('%', ' percent')
          .replace('$', ' dollar ')
          .replace('₹', ' rupee ')
          .replace('€', ' euro ')
          .replace('@', ' at ')
          .replace('[math]', ''))

    # Replace numeric patterns
    q = (q.replace(',000,000,000 ', 'b ')
          .replace(',000,000 ', 'm ')
          .replace(',000 ', 'k '))
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Expand contractions
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not",
        "can't've": "can not have", "'cause": "because",
        "could've": "could have", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
        "he's": "he is", "i'm": "i am", "it's": "it is", "let's": "let us",
        "she's": "she is", "that's": "that is", "there's": "there is",
        "they're": "they are", "we're": "we are", "what's": "what is",
        "who's": "who is", "won't": "will not", "wouldn't": "would not",
        "you're": "you are", "you've": "you have"
    }

    q = ' '.join([contractions.get(word, word) for word in q.split()])

    # Remove HTML and punctuation
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()

    return q


# ==========================
# Query Point Creation
# ==========================
def query_point_creator(q1, q2):
    input_query = []

    # Preprocess
    q1, q2 = preprocess(q1), preprocess(q2)

    # Basic stats
    input_query += [
        len(q1), len(q2),
        len(q1.split()), len(q2.split()),
        test_common_words(q1, q2),
        test_total_words(q1, q2)
    ]
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # Token, length, and fuzzy features
    input_query.extend(test_fetch_token_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    # TF-IDF features
    q1_bow = tfidf.transform([q1]).toarray()
    q2_bow = tfidf.transform([q2]).toarray()

    # Combine all features
    return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))
