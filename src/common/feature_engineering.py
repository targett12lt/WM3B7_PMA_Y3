from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tf_idf(training_data, test_data):
    '''Utilises Sklearn's TdidfVectorizer to create a TD-IDF matrix for the documents provided.

    Inputs:
    * training_data - this should be the column either the cleaned review or the original review from the DF
    * test_data - this should be the column either the cleaned review or the original review from the DF

    Outputs:
    * Vect_Training - Dataframe containing TD-IDF Matrix for the training data.
    * Vect_Testing - Dataframe containing TD-IDF Matrix for the testing/validation data.
    '''
    # Creating Vectorizer object to call:
    TFIDF_Vectorizer = TfidfVectorizer(use_idf=True)
    
    # Vectorized objects:
    Vect_Training = TFIDF_Vectorizer.fit_transform(training_data)
    Vect_Testing = TFIDF_Vectorizer.transform(test_data)
    
    # Printing information about datasets:
    print('TD-IDF Shape for Training data: ', Vect_Training.shape)
    print('TD-IDF Shape for Testing data: ', Vect_Testing.shape)

    # General Information:
    print('IDF Info for Vectorizer: ', TFIDF_Vectorizer.idf_, '\n')

    return Vect_Training, Vect_Testing

def BagOfWords(training_data, test_data):
    '''Utilises Sklearn's 'CountVectorizer' function to convert training and 
    validation/testing data into 'BOW' ready for use by a model.
    
    Inputs: 
    * training_data - this should be the column either the cleaned review or the original review from the DF
    * test_data - this should be the column either the cleaned review or the original review from the DF

    Outputs:
    * BOW_Training - Dataframe containing Bag Of Words for the Training dataframe provided
    * BOW_Testing - Dataframe containing Bag Of Words for the Testing/Validation dataframe provided
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
    '''Allows the user to be able to input the n-value and returns n_grams of ONLY that value
    
    Inputs:
    * n_value (int) - The N-Gram value that should be used when creating a BOW.
    * training_data - this should be the column either the cleaned review or the original review from the DF
    * test_data - this should be the column either the cleaned review or the original review from the DF

    Outputs:
    * ngram_Training - Dataframe containing ngrams for the Training dataframe provided
    * ngram_Testing - Dataframe containing ngrams for the Testing/Validation dataframe provided
    '''
    # 'binary = False' means that the vocabulary vector is filled with term-frequency:
    NGram_Vectorizer = CountVectorizer(binary=False, ngram_range=(n_value, n_value)) 

    ngram_Training = NGram_Vectorizer.fit_transform(training_data)
    ngram_Testing = NGram_Vectorizer.transform(test_data)

    # Information about Training BOW:
    print('Shape of Sparse Matrix:', ngram_Training.shape)
    print('Amount of Non-Zero occurences: ', ngram_Training.nnz)
    print('Sparsity of matrix: ', (100.0 * ngram_Training.nnz / (ngram_Training.shape[0] * ngram_Training.shape[1])))

    # Information about Testing BOW:
    print('\nShape of Sparse Matrix:', ngram_Testing.shape)
    print('Amount of Non-Zero occurences: ', ngram_Testing.nnz)
    print('Sparsity of matrix: ', (100.0 * ngram_Testing.nnz / (ngram_Testing.shape[0] * ngram_Testing.shape[1])))

    return ngram_Training, ngram_Testing

