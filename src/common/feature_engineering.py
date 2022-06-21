from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer



from statistics import mean
import pandas as pd


import nltk
from nltk.sentiment import SentimentAnalyzer

sia = SentimentAnalyzer()


# positive_words = [word for word,
#     nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
# )]
# positive_fd = nltk.FreqDist(positive_words)
# top_100_positive = {word for word, count in positive_fd.most_common(100)}


# Feature engineering = process of creating features for the ML models to use for classification models

# NEEDS TO INCLUDE:
# *N-grams
# * TF-IDF
# *Bag of words approach - a way of extracting features from text for use in modelling, such as ML Algorthms
#   - It describes the occurence of words in a document

def tf_idf(dataset):
    'Utilises the TF-IDF Feature Engineering approach'
    print('dataset:', dataset)
    tfIdfVectorizer= TfidfVectorizer(use_idf=True)
    for review in dataset:
        tfIdf = tfIdfVectorizer.fit_transform(review)
        df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
        df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(25))
    print(tfIdf.shape)
    print(tfIdfVectorizer.get_feature_names_out())
    print(len(tfIdfVectorizer.get_feature_names_out()))

    # tf_idf_transformer = TfidfTransformer(use_idf=True)
    # count_vectorizer = CountVectorizer()
    
    # wordCount = count_vectorizer.fit_transform(dataset)
    # new_tf_idf = tf_idf_transformer.fit_transform(wordCount)
    # df = pd.DataFrame(new_tf_idf[0].T.todense(), index=count_vectorizer.get_feature_names(), columns=["TF-IDF"])
    # df = df.sort_values('TF-IDF', ascending=False)
    # print (df.head(25))



def tf(bagOfWords, ):
    '''Utilises the term frequency (TF) - weighting method
    
    Inputs: 
    * 'bagOfWords' - Bag of Words output from function
    * 'NumberOfWords' - The number of unique words in the 'document'

    Outputs:
    * Dictionary containing term frequency

    '''
    tf_Dictionary = {}
    Length_BagOfWords = len(bagOfWords)

def n_gram(n_value):
    'Allows the user to be able to input the n-value'

def FrequencyDistribution():
    '''Utilises nltk's frequency distribution function'''
    print("Hi from the FD Function")

def bagOfWords(documents):
    '''Utilises SkLean's Count Vectoriser function to implement
    a "bag of words" method
    
    Function assumes all sentances have been tokenized as part of preprocessing
    '''
    vectorizer = CountVectorizer()  # Initializing 'CountVectorizer'
    x = vectorizer.fit_transform(documents.values)
    
    return vectorizer.get_feature_names(), x.toarray()

