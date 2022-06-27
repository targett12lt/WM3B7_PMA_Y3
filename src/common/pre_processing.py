# General modules:
import os
import re

# For data:
import pandas as pd
# import pickle  # To save the model

# For NLP/Sentinent Analysis:
import nltk
# import sklearn


# Downloading 'stopwords' and 'punk' so script can work correctly
# Try and make this a try and except statement (checks if PC has them downloaded already, if not then downloads them?)
nltk.download('stopwords')  # Download 'stopwords' for nltk library
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Defining 'english' stopwords for nltk
stop_words = list(set(nltk.corpus.stopwords.words('english')))


# ''' THINGS THAT NEED TO BE INCLUDED:
# * Import the data - DONE
# * Make all text lowercase - DONE
# * Remove all punctuation - DONE
# * Tokenization - DONE
# * Normalizing words (condensing different forms of the same word into a single form): Two major methods -> Stemming or Lemmaziation - DONE
# * Removing Stop Words - DONE
# * Remove low frequency words? This can probabably be achieved using TF or TF-IDF
# '''

# def getWords(CleanedList):
#     '''Takes a list of 'cleaned' words and returns a string containing all words'''
#     for tokens in CleanedList:
#         for token in tokens:
#             yield token

def cleanText(text):
    '''Cleans the supplied text - splits text into strings provided string is in sentence/paragraph format.
    
    Cleans the text by:
        1. Removing special characters & punctuation from strings
        2. Makes all text lowercase
        3. Creates a 'tokenized' list of the sentence
        4. Removes 'stop words' from the sentence
        5. Lemmatizes the string (normalization method to condense all forms of a word into a single representation)

    ADD MORE INFORMATION ABOUT FUNCTION HERE
    '''
    # Creating Lemmatizer required to clean text:
    Lemmatizer = nltk.stem.WordNetLemmatizer() 

    # Cleaning text:
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', text)  # Removing punctuation and special characters
    lowered = cleaned.lower()  # Making all text lowercase
    tokenized_list = nltk.tokenize.word_tokenize(lowered)
    stopped = [w for w in tokenized_list if not w in stop_words]  # Removing stop words using NLTK's English stop words
    lemmatized = [Lemmatizer.lemmatize(w, pos='v') for w in stopped]

    # Converting list back into string:
    cleaned_string = ' '.join(lemmatized)

    # print(cleaned_string)
    return cleaned_string

def importData(dataDirectory):
    '''Imports the data from the supplied directory "dataDirectory" and returns
    data in a list format, ensuring data is "shuffled" so positive and negative
    reviews aren't split obviously.'''
    df = pd.DataFrame(columns=['Review', 'PositiveReview', 'NegativeReview', 'Sentiment'])
    reviews = []  # Creating a basic structure to store data in

    for label in ["pos", "neg"]:
        labeled_directory = f"{dataDirectory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")  # Removing HTML formatting and replacing with Python formatting
                    if text.strip():  # Removing whitespace from start & end of strings
                        df.loc[df.shape[0]] = [text, "pos" == label, "neg" == label, None]  # Without cleaning
                        

    # Shuffling list 'reviews' so the same type are not all next to each other
    # df.sample(frac=1, random_state=1).reset_index()

    return df


def save_to_pkl(dataframe, df_name):
    '''Saves the supplied dataframe in PKL format in the current working directory'''
    # current_working_directory = os.getcwd()
    dataframe.to_pickle(df_name)

def read_from_pkl(pkl_location):
    '''Creates and returns a Pandas Dataframe from a specified PKL file'''
    
    df = pd.read_pickle(pkl_location)
    return df

def add_cleaned_column(dataframe, original_col_name, new_col_name, clean=True):
    '''Adds a new column containing all the reviews but not cleaned'''
    # dataframe[new_col_name] = dataframe[original_col_name]
    # if clean:
    #     dataframe[new_col_name].apply(cleanText)
    dataframe[new_col_name] = dataframe[original_col_name].apply(cleanText)
