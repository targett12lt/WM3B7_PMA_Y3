# General modules:
import os
import re

# For data:
import pandas as pd

# For NLP/Sentinent Analysis:
import nltk

# Downloading required libraries from NLTK so script can work as required:
print('Downloading required NLTK Resources... \n\n')
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4'])

# Defining 'english' stopwords for nltk
stop_words = list(set(nltk.corpus.stopwords.words('english')))


def cleanText(text):
    '''Cleans the supplied text - splits text into strings provided string is
    in sentence/paragraph format.

    Cleans the text by:
        1. Removing special characters & punctuation from strings
        2. Makes all text lowercase
        3. Creates a 'tokenized' list of the sentence
        4. Removes 'stop words' from the sentence
        5. Lemmatizes the string (normalization method to condense all forms
           of a word into a single representation)

    INPUTS:
    * text - Document/Review to be cleaned in 'str' format

    OUTPUTS:
    * cleaned_string - Cleaned Document/Review in 'str' format
    '''
    # Creating Lemmatizer required to clean text:
    Lemmatizer = nltk.stem.WordNetLemmatizer()

    # Cleaning text:
    # Removing punctuation & special characters:
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', text)
    lowered = cleaned.lower()  # Making all text lowercase
    tokenized_list = nltk.tokenize.word_tokenize(lowered)
    # Removing stop words using NLTK's English stop words:
    stopped = [w for w in tokenized_list if not w in stop_words]
    lemmatized = [Lemmatizer.lemmatize(w, pos='v') for w in stopped]

    # Converting list back into string:
    cleaned_string = ' '.join(lemmatized)

    return cleaned_string


def importData(dataDirectory):
    '''Imports the data from the supplied directory "dataDirectory" and returns
    data in a list format, ensuring data is "shuffled" so positive and negative
    reviews aren't split obviously.

    INPUTS:
    * dataDirectory - Directory that stores all the data to be imported into
    dataframe

    OUTPUTS:
    * df - Dataframe containing all data from specified data directory
    '''
    df = pd.DataFrame(columns=['Review', 'Sentiment'])

    for label in ["pos", "neg"]:
        labeled_directory = f"{dataDirectory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    text = f.read()
                    # Removing HTML formatting & replacing with Python equiv:
                    text = text.replace("<br />", "\n\n")
                    # Removing whitespace from start & end of strings:
                    if text.strip():
                        if label == 'pos':
                            df.loc[df.shape[0]] = [text, 1]  # Without cleaning
                        else:
                            df.loc[df.shape[0]] = [text, 0]  # Without cleaning

    return df


def save_to_pkl(dataframe, df_name):
    '''Saves the supplied dataframe in PKL format in the current working
    directory. Function does NOT return anything!

    INPUTS:
    * dataframe - The Pandas Dataframe that you would like saving as a '.pkl'
    * df_name - What you'd like the '.pkl' to be called
    '''
    dataframe.to_pickle(df_name)


def read_from_pkl(pkl_name):
    '''Creates and returns a Pandas Dataframe from a specified PKL file.
    Function does NOT return anything!

    INPUTS:
    * pkl_name - Name of the '.pkl' stored in the CWD
    '''
    df = pd.read_pickle(pkl_name)
    return df


def add_cleaned_column(dataframe, original_col_name, new_col_name):
    '''Adds a new column containing all the reviews AFTER they have been
    cleaned. Function does NOT return anything!

    INPUTS:
    * dataframe - The dataframe that you'd like the new column in
    * original_col_name - The column which contains the 'Reviews' in the df.
    * new_col_name - The name of the new 'clean' column
    '''
    dataframe[new_col_name] = dataframe[original_col_name].apply(cleanText)
