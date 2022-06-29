import pandas as pd
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Feature engineering = process of creating features for the ML models to use for classification models

# NEEDS TO INCLUDE:
# *N-grams - WORKING
# * TF-IDF - WORKING
# *Bag of words approach - a way of extracting features from text for use in modelling, such as ML Algorthms
#   - It describes the occurence of words in a document
# BOW - WORKING

def tf_idf(training_data, test_data):
    '''
    INFORMATION ABOUT FUNCTION HERE

    Inputs:
    * 'training_data' - 
    * 'test_data' - 

    Outputs:
    * Vect_Training - 
    * Vect_Testing -
    * featureNames - 
    '''
    # Creating Vectorizer object to call:
    TFIDF_Vectorizer = TfidfVectorizer(use_idf=True)
    
    # Vectorized objects:
    Vect_Training = TFIDF_Vectorizer.fit_transform(training_data)
    Vect_Testing = TFIDF_Vectorizer.transform(test_data)
    
    # Printing information about datasets:
    print('Vect_Training: ', Vect_Training.shape)
    print('Vect_Testing: ', Vect_Testing.shape)

    # General Information:
    print('IDF Info for Vectorizer: ', TFIDF_Vectorizer.idf_)

    return Vect_Training, Vect_Testing

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

def n_gram(n_value: int, training_data, test_data):
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

