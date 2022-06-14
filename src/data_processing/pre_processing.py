import os
import re
import random
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Downloading 'stopwords' and 'punk' so script can work correctly
nltk.download('stopwords')  # Download 'stopwords' for nltk library
nltk.download('punkt')

# Defining 'english' stopwords for nltk
stop_words = list(set(stopwords.words('english')))


# import sklearn
# import pickle  # To save the model

# ''' THINGS THAT NEED TO BE INCLUDED:
# * Import the data - DONE
# * Make all text lowercase - DONE
# * Remove all punctuation - DONE
# * Remove any emojis?? If there are any?
# * Tokenization - DONE
# * Normalizing words (condensing different forms of the same word into a single form): Two major methods -> Stemming or Lemmaziation
# * BoW Model
# * Removing Stop Words - DONE
# * Remove low frequency words 
# '''




# Getting file path to data:
os.chdir('....')  # Changing the working directory from script directory
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')  # Specifying base data directory

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')

# print('base_folder_data:', base_folder_data)  # FOR DEBUGGING PURPOSES


def cleanText(text):
    '''Cleans the supplied text
    
    ADD MORE INFORMATION ABOUT FUNCTION HERE
    '''
    
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', text)  # Removing punctuation and special characters
    lowered = cleaned.lower()  # Making all text lowercase
    tokenized = word_tokenize(lowered)
    stopped = [w for w in tokenized if not w in stop_words]  # Removing stop words using NLTK's English stop words

    return stopped

def importData(dataDirectory):
    '''Imports the data from the supplied directory "dataDirectory" and returns
    data in a list format, ensuring data is "shuffled" so positive and negative
    reviews aren't split obviously.'''
    reviews = []  # Creating a basic structure to store data in

    for label in ["pos", "neg"]:
        labeled_directory = f"{dataDirectory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding="utf8") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")  # Removing HTML formatting and replacing with Python formatting
                    if text.strip():  # Removing whitespace from start & end of strings
                        labels = {
                            "Review Category": {
                                "Positive": "pos" == label,
                                "Negative": "neg" == label
                            }
                        }
                        reviews.append((cleanText(text), labels))

    # Shuffling list 'reviews' so the same type are not all next to each other
    random.shuffle(reviews)

    return reviews


training_data_set = importData(train_data)
print(training_data_set)

