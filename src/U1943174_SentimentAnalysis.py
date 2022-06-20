import common
import models

import os

# Getting file path to data:
os.chdir('....')  # Changing the working directory from script directory
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')  # Specifying base data directory

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')

# print('base_folder_data:', base_folder_data)  # FOR DEBUGGING PURPOSES

# If running from scratch and creating PKL:
# training_data = common.importData(train_data)
# common.save_to_pkl(training_data, 'PreProcessedTrainingData.pkl')  # Saving data to pkl so it can be opened quickly

# Importing data from PKL file to make debugging/development quicker:
training_data = common.read_from_pkl('PreProcessedTrainingData.pkl')

# print(training_data.head(30))  # FOR DEBUGGING PURPOSES

# TF-IDF Approach:
# TF_IDF_DataFrame = common.tf_idf(training_data['Review'])

stitched_strings = training_data['Review'].apply(lambda x: ' '.join([word for word in x]))
bag_of_words, documentVectors = common.bagOfWords(stitched_strings)
print(bag_of_words)
print(documentVectors)

