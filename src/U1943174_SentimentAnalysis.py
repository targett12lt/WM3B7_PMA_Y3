import common
import models

import os
# import pandas as pd


# Getting file path to data:
os.chdir('....')  # Changing the working directory from script directory
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')  # Specifying base data directory

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')

# print('base_folder_data:', base_folder_data)  # FOR DEBUGGING PURPOSES

# Creating Pandas DF:
# training_df = pd.DataFrame(columns=['Review', 'PositiveReview', 'NegativeReview', 'Sentiment'])

training_data = common.importData(train_data)
print(training_data)