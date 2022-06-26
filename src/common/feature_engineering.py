from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from . import pre_processing


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

def tf_idf(BOW):
    '''Utilises the TF-IDF Feature Engineering approach
    
    REQUIRES: BOW INPUT

    '''
    all_words = getWords(BOW)

    print('All words == ', all_words)

    TFIDF_Transformer = TfidfTransformer().fit(all_words)
    print(TFIDF_Transformer.idf_)



def tf(dataframe):
    '''Utilises the term frequency (TF) - weighting method
    
    Inputs: 
    * 'bagOfWords' - Bag of Words output from function
    * 'NumberOfWords' - The number of unique words in the 'document'

    Outputs:
    * Dictionary containing term frequency

    '''

    BOW_Transformer = CountVectorizer(analyzer = pre_processing.cleanText).fit(dataframe['Review'])

    bag_of_words_review = BOW_Transformer.transform(dataframe['Review'])

    print('Shape of Sparse Matrix:', bag_of_words_review.shape)
    print('Amount of Non-Zero occurences: ', bag_of_words_review.nnz)
    print('Sparsity of matrix: ', (100.0 * bag_of_words_review.nnz / (bag_of_words_review.shape[0] * bag_of_words_review.shape[1])))
    # Need to add returns stuff here

    return bag_of_words_review
    

def n_gram(n_value):
    'Allows the user to be able to input the n-value'

def getWords(CleanedList):
    '''Takes a list of 'cleaned' words and returns a string containing all words'''
    for tokens in CleanedList:
        for token in tokens:
            yield token

def FrequencyDistribution(Reviews):
    '''Utilises nltk's frequency distribution function
    
    Tells you how many times a specific word appears in a given text

    '''    
    all_words = getWords(Reviews)

    fd = nltk.FreqDist(all_words)
    most_common = fd.most_common(25)
    print(fd)
    print('Most common: ', most_common, '\n\n\n\n')
    print('Tabulated: ', fd.tabulate(5))

def word_count():
    '''Counts the number of words in a document and returns the value'''

    print('Hello world')

