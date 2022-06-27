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

################################ IMPORTING DATA ################################

# If running from scratch and creating PKL:
# training_data = common.importData(train_data)
# common.save_to_pkl(training_data, 'TrainingData.pkl')  # Saving data to pkl so it can be opened quickly

# Importing data from PKL file to make debugging/development quicker:
training_data = common.read_from_pkl('TrainingData.pkl')

################################ CLEANING DATA #################################

common.add_cleaned_column(training_data, 'Review', 'CleanedReview')

print('Cleaned Pandas DF:')
print(training_data)

############################### DATA EXPLORATION ###############################

# NEED TO PUT SOME DATA EXPLORATION/VISUALISATION HERE
common.visualise_sentiment_type(training_data)

############################## FEATURE ENGINEERING #############################

bag_of_words = common.BagOfWords(training_data)

# common.FrequencyDistribution(training_data['CleanedReview'])

common.tf_idf(training_data)  # Uses previous bag of words to avoid duplication of process (resulting in poorer compuational performance)

print(training_data)

