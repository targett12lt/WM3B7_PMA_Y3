import os
import random

import nltk
from nltk.corpus import stopwords
# import sklearn
# import pickle  # To save the model

# Getting file path to data:
os.chdir('....')
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')
print('base_folder_data:', base_folder_data)


# files_pos = os.listdir(os.path.join(train_data, 'pos'))
# files_pos = [open(os.path.join(train_data, 'pos')+f, 'r').read() for f in files_pos]

# print('Files pos: ', files_pos)

# files_neg = os.listdir(os.path.join(train_data, 'neg'))
# files_neg = [open('train/neg/'+f, 'r').read() for f in files_neg]

def importData(dataDirectory):
    '''Imports the data from the supplied directory "dataDirectory" and returns
    data in a list format, ensuring data is "shuffled" so positive and negative
    reviews aren't split obviously.'''
    reviews = []  # Creating a basic structure to store data in

    for label in ["pos", "neg"]:
        labeled_directory = f"{train_data}/{label}"
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
                        reviews.append((text, labels))

    # Shuffling list 'reviews' so the same type are not all next to each other
    random.shuffle(reviews)

    return reviews  # Returning list of reviews so it can be used to train a model


training_data_set = importData(train_data)


