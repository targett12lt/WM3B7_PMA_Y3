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
training_data = common.importData(train_data)
common.save_to_pkl(training_data, 'TrainingData.pkl')  # Saving data to pkl so it can be opened quickly

# Importing data from PKL file to make debugging/development quicker:
# training_data = common.read_from_pkl('TrainingData.pkl')

# print(training_data.head(30))  # FOR DEBUGGING PURPOSES

# TF-IDF Approach:
# TF_IDF_DataFrame = common.tf_idf(training_data['Review'])

# stitched_strings = training_data['Review'].apply(lambda x: ' '.join([word for word in x]))
# word_list, bagOfWordsVector = common.bagOfWords(stitched_strings)
# print(word_list)
# print(bagOfWordsVector)

common.add_cleaned_column(training_data, 'Review', 'CleanedReview')

print('Cleaned Pandas DF:')

print(training_data)

bag_of_words = common.tf(training_data)

common.FrequencyDistribution(training_data['CleanedReview'])

common.tf_idf(bag_of_words)  # Uses previous bag of words to avoid duplication of process (resulting in poorer compuational performance)

print(training_data)

