from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from . import pre_processing


from statistics import mean
import pandas as pd


import nltk
from nltk.sentiment import SentimentAnalyzer

sia = SentimentAnalyzer()

# Feature engineering = process of creating features for the ML models to use for classification models

# NEEDS TO INCLUDE:
# *N-grams
# * TF-IDF
# *Bag of words approach - a way of extracting features from text for use in modelling, such as ML Algorthms
#   - It describes the occurence of words in a document

def tf_idf(dataframe):
    '''Utilises the TF-IDF Feature Engineering approach
    
    REQUIRES: BOW INPUT

    '''
    # all_words = getWords(BOW)

    # print('All words == ', all_words)

    TFIDF_Vectorizer = TfidfVectorizer(stop_words=pre_processing.stop_words)
    X_val = TFIDF_Vectorizer.fit_transform(dataframe['Review'])
    Y_Val = TFIDF_Vectorizer.fit_transform(dataframe['Review'])
    featureNames = TFIDF_Vectorizer.get_feature_names()
    print(TFIDF_Vectorizer.idf_)

    return X_val, Y_Val, featureNames

def DocumentTermMatrix(dataframe, X_Vals, Y_Vals, featureNames):
    '''
    INPUTS:
    * X_val - 
    * Y_val - 
    * featureNames - 

    Information about function here

    OUTPUTS:
    * something
    '''
    DocTerm_DF = pd.DataFrame(X_Vals.toarray(), columns=featureNames)
    DocTerm_DF['Sentiment'] = LabelEncoder().fit_transform(dataframe['Sentiment'])

    # Printing information about 'DocTerm_DF' dataframe:
    print(DocTerm_DF.shape)
    print(DocTerm_DF.head(10))

    return DocTerm_DF


def BagOfWords(training_data, test_data):
    # No need to create Term Frequency and BOW methods, as both are very similar,
    # BOW creates vectors for 
    
    '''Utilises the term frequency (TF) - weighting method
    
    Inputs: 
    * 'training_data' - this should be the column either the cleaned review or the original review from the DF
    * 'test_data' - this should be the column either the cleaned review or the original review from the DF

    Outputs:
    * Dictionary containing term frequency

    '''
    # 'binary = False' means that the vocabulary vector is filled with term-frequency:
    BOW_Vectorizer = CountVectorizer(binary=False)

    BOW_Training = BOW_Vectorizer.fit_transform(training_data)
    BOW_Testing = BOW_Vectorizer.transform(test_data)

    # Information about Training BOW:
    print('Shape of Sparse Matrix:', BOW_Training.shape)
    print('Amount of Non-Zero occurences: ', BOW_Training.nnz)
    print('Sparsity of matrix: ', (100.0 * BOW_Training.nnz / (BOW_Training.shape[0] * BOW_Training.shape[1])))

    # Information about Testing BOW:
    print('\nShape of Sparse Matrix:', BOW_Testing.shape)
    print('Amount of Non-Zero occurences: ', BOW_Testing.nnz)
    print('Sparsity of matrix: ', (100.0 * BOW_Testing.nnz / (BOW_Testing.shape[0] * BOW_Testing.shape[1])))

    return BOW_Training, BOW_Testing  

def n_gram(n_value, training_data, test_data):
    'Allows the user to be able to input the n-value and returns n_grams of ONLY that value'
    # 'binary = False' means that the vocabulary vector is filled with term-frequency:
    BOW_Vectorizer = CountVectorizer(binary=False, ngram_range=(n_value, n_value))

    ngram_Training = BOW_Vectorizer.fit_transform(training_data)
    ngram_Testing = BOW_Vectorizer.transform(test_data)

    # Information about Training BOW:
    print('Shape of Sparse Matrix:', ngram_Training.shape)
    print('Amount of Non-Zero occurences: ', ngram_Training.nnz)
    print('Sparsity of matrix: ', (100.0 * ngram_Training.nnz / (ngram_Training.shape[0] * ngram_Training.shape[1])))

    # Information about Testing BOW:
    print('\nShape of Sparse Matrix:', ngram_Testing.shape)
    print('Amount of Non-Zero occurences: ', ngram_Testing.nnz)
    print('Sparsity of matrix: ', (100.0 * ngram_Testing.nnz / (ngram_Testing.shape[0] * ngram_Testing.shape[1])))

    return ngram_Training, ngram_Testing  

def FrequencyDistribution(Reviews):
    # NOT THE SAME AS TERM FREQUENCY, as it is the 'Frequency Distribution'

    '''Utilises nltk's frequency distribution function
    
    Tells you how many times a specific word appears in a given text

    '''  
    print('Type(Reviews): ', type(Reviews))
    print(Reviews)  
    all_words = Reviews.split()

    fd = nltk.FreqDist(all_words)
    most_common = fd.most_common(25)
    print(fd)
    print('Most common: ', most_common, '\n\n\n\n')
    print('Tabulated: ', fd.tabulate(5))

def word_count(dataframe):
    '''Counts the number of words in a document and returns the value'''
    dataframe['WordCount']  = dataframe['Review'].apply(lambda val: len(str(val).split(' ')))

    print('Hello world')

