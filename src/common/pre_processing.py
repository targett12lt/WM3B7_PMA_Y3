# General modules:
import os
import re
import random

# For data:
# import pandas  # Getting DLL error when importing at the moment
# import pickle  # To save the model

# For NLP/Sentinent Analysis:
import nltk
import pandas as pd
import numpy as np
import numpy
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


def cleanText(text):
    '''Cleans the supplied text - splits text into strings provided string is in sentence/paragraph format.
    
    Cleans the text by:
        * Removing special characters & punctuation from strings
        * Makes all text lowercase
        * Creates a 'tokenized' list of the sentence
        * Removes 'stop words' from the sentence
        * Lemmatizes the string (normalization method to condense all forms of a word into a single representation)

    ADD MORE INFORMATION ABOUT FUNCTION HERE
    '''
    # Creating Lemmatizer required to clean text:
    Lemmatizer = nltk.stem.WordNetLemmatizer() 

    cleaned = re.sub(r'[^(a-zA-Z)\s]','', text)  # Removing punctuation and special characters
    lowered = cleaned.lower()  # Making all text lowercase
    tokenized_list = nltk.tokenize.word_tokenize(lowered)
    stopped = [w for w in tokenized_list if not w in stop_words]  # Removing stop words using NLTK's English stop words
    lemmatized = [Lemmatizer.lemmatize(w, pos='v') for w in stopped]

    return lemmatized

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
                        # labels = {
                        #     "Review Category": {
                        #         "Positive": "pos" == label,
                        #         "Negative": "neg" == label
                        #     }
                        # }

                        # df.loc[df.shape[0]] = [text, "pos" == label, "neg" == label, None]  # Without cleaning
                        df.loc[df.shape[0]] = [cleanText(text), "pos" == label, "neg" == label, None]
                        

    # Shuffling list 'reviews' so the same type are not all next to each other
    # df.sample(frac=1, random_state=1).reset_index()

    return df


