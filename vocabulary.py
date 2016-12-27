from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(input = u'content',
                             analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None, \
                             token_pattern = ur',|\b\w+\b', \
                             max_features = 15448) 

import pandas as pd

def generateVocab(filename):
    """
    Here, the argument is the name of the file which contains the text data.
    It is a tab separated file sentences are under the column header 'sentences' 
    """
    df = pd.read_csv(filename, sep = '\t')
    sentences = []
    for i in df['sentence']:
        sentences.append(i)
    vectorizer.fit(sentences)
    return vectorizer.vocabulary_