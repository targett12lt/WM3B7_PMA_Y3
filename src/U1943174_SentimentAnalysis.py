# Importing packages:
import common
import models

# Importing external libraries:
import numpy as np
import os
import warnings

# Sklearn Imports:
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# NEED TO ADD CROSS VALIDATION 

# X Add comment here about why this is needed:
warnings.filterwarnings('ignore')

# Getting file path to data:
os.chdir('....')  # Changing the working directory from script directory
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')  # Specifying base data directory

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')

model = MultinomialNB()
print(model.get_params().keys())

################################ IMPORTING DATA ################################
# Logic to choose data import method:
import_data = input('Would you like to import the data from the file structure (1)'
' or the ".pkl" (2)?\n\nPlease enter the number of your preferred option followed'
' by "Enter": ')

# If running from scratch and creating PKL:
if '1' in import_data:
    # Training Data:
    print('Importing Training Data to DataFrame...')
    training_data = common.importData(train_data)
    common.save_to_pkl(training_data, 'TrainingData.pkl')  # Saving data to pkl so it can be opened quickly

    # Test Data:
    print('Importing Test Data to DataFrame...')
    test_data = common.importData(test_data)
    common.save_to_pkl(test_data, 'TestData.pkl')  # Saving data to pkl so it can be opened quickly

else:
    # Importing data from PKL file to make debugging/development quicker:
    training_data = common.read_from_pkl('TrainingData.pkl')
    test_data = common.read_from_pkl('TestData.pkl')

################################ CLEANING DATA #################################

common.add_cleaned_column(training_data, 'Review', 'CleanedReview')
common.add_cleaned_column(test_data, 'Review', 'CleanedReview')

print()
print('Cleaned Pandas DF:\n', training_data)

############################### DATA EXPLORATION ###############################

# NEED TO PUT SOME DATA EXPLORATION/VISUALISATION HERE
# common.visualise_sentiment_type(training_data)

##################### SPLITTING DATA INTO TRAIN & VALIDATE #####################

X_train, X_validate, Y_train, Y_validate = train_test_split(training_data.CleanedReview, training_data.Sentiment, test_size=0.15, random_state=42)

Y_train = Y_train.astype('int')
Y_validate = Y_validate.astype('int')

# Defining reviews that haven't been cleaned:
train_reviews_original = training_data.Review
test_reviews_original = test_data.Review

############################## FEATURE ENGINEERING #############################

'''The data should be split here before feature engineering begins ******************LOOOOOOK AT ME************************'''

# BAG OF WORDS (Unigram) with no cleaning:
Dirty_BOW_Training, Dirty_BOW_Testing =  common.BagOfWords(train_reviews_original, test_reviews_original)

# BAG OF WORDS (Unigram):
BOW_Training, BOW_Testing = common.BagOfWords(X_train, X_validate)

# TF-IDF:
Vect_Training, Vect_Testing = common.tf_idf(X_train, X_validate)

# Bigram:
Bigram_Training, Bigram_Testing = common.n_gram(2, X_train, X_validate)

# Trigram:
Trigram_Training, Trigram_Testing = common.n_gram(3, X_train, X_validate)

######################## USING FEATURES TO TRAIN MODELS ########################

######################## LOGISTIC REGRESSION ########################

# LR With BOW With DIRTY Data:
models.logRegression(Dirty_BOW_Training, Y_train, Dirty_BOW_Testing, Y_validate, 'BOW Dirty')

####### Logistic Regssion Model with BOW's:
models.logRegression(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW')

####### Logistic Regssion Model with TF-IDF:
models.logRegression(Vect_Training, Y_train, Vect_Testing, Y_validate, 'TF-IDF')

####### Logistic Regssion Model with Bigrams:
models.logRegression(Bigram_Training, Y_train, Bigram_Testing, Y_validate, 'Bigram')

####### Logistic Regssion Model with Trigrams:
models.logRegression(Trigram_Training, Y_train, Trigram_Testing, Y_validate, 'Trigram')

##################### MULTINOMINAL NAIVE BAYES ######################

###### Multinominal Naive Bayes Model with DIRTY Data:
models.MultiNaiveBayes(Dirty_BOW_Training, Y_train, Dirty_BOW_Testing, Y_validate, 'BOW Dirty')

####### Multinominal Naive Bayes Model with BOW's:
models.MultiNaiveBayes(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW')

####### Multinominal Naive Bayes Model with TF-IDF:
models.MultiNaiveBayes(Vect_Training, Y_train, Vect_Testing, Y_validate, 'TF-IDF')

####### Multinominal Naive Bayes Model with Bigrams:
models.MultiNaiveBayes(Bigram_Training, Y_train, Bigram_Testing, Y_validate, 'Bigram')

####### Multinominal Naive Bayes Model with Trigrams:
models.MultiNaiveBayes(Trigram_Training, Y_train, Trigram_Testing, Y_validate, 'Trigram')

############################ LINEAR SVC #############################

###### Linear SVC Model with DIRTY Data:
models.LinSVC(Dirty_BOW_Training, Y_train, Dirty_BOW_Testing, Y_validate, 'BOW Dirty')

####### Linear SVC with Bag of Words:
models.LinSVC(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW')

####### Linear SVC Model with TF-IDF:
models.LinSVC(Vect_Training, Y_train, Vect_Testing, Y_validate, 'TF-IDF')

####### Linear SVC Model with Bigrams:
models.LinSVC(Bigram_Training, Y_train, Bigram_Testing, Y_validate, 'Bigram')

####### Linear SVC Model with Trigrams:
models.LinSVC(Trigram_Training, Y_train, Trigram_Testing, Y_validate, 'Trigram')

############ OPTIMISING HYPERPARAMETERS FOR LOGISTIC REGRESSION ###############

# Creating Logistic Regression Model:
model = LogisticRegression()

# Defining 'parameter grid' for Logisitic Regression:
parameters = {
    'penalty' : ['l1','l2'],  # Regularization of the data (not all solvers support this)
    'C'       : [100, 10, 1.0, 0.1, 0.01],
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
# Defining the Grid Search function with the desired parameters:
clf = GridSearchCV(model, param_grid = parameters, scoring = 'accuracy', cv = 10)
clf.fit(Vect_Training, Y_train)

# Printing the best parameters:
print("Best score achieved using %s" % (clf.best_params_))

# Testing the new one on the validation data:
models.logRegression(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW', **clf.best_params_)

########### OPTIMISING HYPERPARAMETERS FOR MULTINOMINAL NAIVE BAYES ############

# Creating Multinominal Naive Bayes:
model = MultinomialNB()

# Defining 'parameter grid' for MultinominalNB:
params_NB = {'alpha': np.logspace(0,-9, num=100)}
# High alpha -> Underfitting
# Low alpha -> Overfitting

# Defining the Grid Search function with the desired parameters:
clf_nb = GridSearchCV(model, param_grid = params_NB, scoring = 'accuracy', cv = 10)

clf_nb.fit(Vect_Training, Y_train)

# Printing the best parameters:
print("Best score achieved using %s" % (clf_nb.best_params_))

# Testing the new one on the validation data:
models.MultiNaiveBayes(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW', **clf_nb.best_params_)

################## OPTIMISING HYPERPARAMETERS FOR LINEAR SVC ###################

# Creating Linear SVC:
model_svc = LinearSVC()

# Defining 'parameter grid' for Linear SVC:
parameters_linear_svc = {
    'penalty' : ['l1','l2'],  # Regularization of the data (not all solvers support this)
    'loss': ['hinge', 'squared_hinge'],
    'C'       : [100, 10, 1.0, 0.1, 0.01],
}

# Defining the Grid Search function with the desired parameters:
clf_svc = GridSearchCV(model_svc, param_grid = parameters_linear_svc, scoring = 'accuracy', cv = 10)

clf_svc.fit(Vect_Training, Y_train)

# Printing the best score and parameters:
print("Best score achieved using %s" % (clf_svc.best_params_))

# Testing new one:
models.LinSVC(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW', **clf_svc.best_params_)

################## TESTING TUNED ALGORITHMS ON TESTING DATA ####################

'''Need to use the testing data here and output it's efficiency'''

#### Setting variables for test data:

test_reviews = test_data.CleanedReview
test_sentiments = test_data.Sentiment
test_sentiments = test_sentiments.astype('int')

# Measuring model on test dataset using the three models:
models.logRegression(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW', **clf.best_params_)
models.MultiNaiveBayes(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW', **clf_nb.best_params_)
models.LinSVC(BOW_Training, Y_train, BOW_Testing, Y_validate, 'BOW', **clf_svc.best_params_)

