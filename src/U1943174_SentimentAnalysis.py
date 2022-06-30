import numpy as np

# NEED TO ADD CROSS VALIDATION 
from sklearn.model_selection import cross_validate

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB # Suitable for classification with discrete features, can work with td-idf but made for integer counts ideally

import common
import models

import os
import warnings
warnings.filterwarnings('ignore')

# Getting file path to data:
os.chdir('....')  # Changing the working directory from script directory
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')  # Specifying base data directory

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')

# print('base_folder_data:', base_folder_data)  # FOR DEBUGGING PURPOSES

model = MultinomialNB()
print(model.get_params().keys())

################################ IMPORTING DATA ################################
# Logic to choose data import method:
import_data = input('Would you like to import the data from the file structure (1)'
' or the ".pkl" (2)?\n\nPlease enter the number of your preferred option followed'
' by "Enter":')

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

####################### SPLITTING DATA INTO TEST & TRAIN #######################

# X_Train, Y_Train = DocTermMatrix.drop(['Sentiment'], axis = 1), DocTermMatrix['Sentiment']
# X_Test, Y_Test = DocTermMatrixTest.drop(['Sentiment'], axis = 1), DocTermMatrixTest['Sentiment']

# Testing using original data:
# X_Train, Y_Train = training_data.drop(['Sentiment'], axis = 1), training_data['Sentiment']
# X_Test, Y_Test = test_data.drop(['Sentiment'], axis = 1), test_data['Sentiment']

# X_train, X_validate, Y_train, Y_Validate = train_test_split(training_data.CleanedReview, training_data.Sentiment, test_size=0.2, random_state=42)

# X_validate = X_validate.astype('int')
# Y_Validate = Y_Validate.astype('int')

# Setting variables:

# Defining reviews that haven't been cleaned:
train_reviews_original = training_data.Review
test_reviews_original = test_data.Review

# Cleaned vars and sentiments:
train_reviews = training_data.CleanedReview
train_sentiments = training_data.Sentiment
train_sentiments = train_sentiments.astype('int')

test_reviews = test_data.CleanedReview
test_sentiments = test_data.Sentiment
test_sentiments = test_sentiments.astype('int')

############################## FEATURE ENGINEERING #############################


'''The data should be split here before feature engineering begins ******************LOOOOOOK AT ME************************'''

# common.FrequencyDistribution(training_data['CleanedReview'])

# Creating a Document Term Matrix:
# DocTermMatrix = common.DocumentTermMatrix(training_data, TFIDF_Output[0], TFIDF_Output[1], TFIDF_Output[2])
# DocTermMatrixTest = common.DocumentTermMatrix(test_data, TFIDF_Output_Test[0], TFIDF_Output_Test[1], TFIDF_Output_Test[2])

# BAG OF WORDS (Unigram) with no cleaning:
Dirty_BOW_Training, Dirty_BOW_Testing =  common.BagOfWords(train_reviews_original, test_reviews_original)

# BAG OF WORDS (Unigram):
BOW_Training, BOW_Testing = common.BagOfWords(training_data.CleanedReview, test_data.CleanedReview)

# TF-IDF:
Vect_Training, Vect_Testing = common.tf_idf(training_data.CleanedReview, test_data.CleanedReview)

# Bigram:
Bigram_Training, Bigram_Testing = common.n_gram(2, training_data.CleanedReview, test_data.CleanedReview)

# Trigram:
Trigram_Training, Trigram_Testing = common.n_gram(3, training_data.CleanedReview, test_data.CleanedReview)

######################## USING FEATURES TO TRAIN MODELS ########################

######################## LOGISTIC REGRESSION ########################

# LR With BOW With DIRTY Data:
models.logRegression(Dirty_BOW_Training, train_sentiments, Dirty_BOW_Testing, test_sentiments, 'BOW Dirty')

####### Logistic Regssion Model with BOW's:

models.logRegression(BOW_Training, train_sentiments, BOW_Testing, test_sentiments, 'BOW')

####### Logistic Regssion Model with TF-IDF:
models.logRegression(Vect_Training, train_sentiments, Vect_Testing, test_sentiments, 'TF-IDF')

####### Logistic Regssion Model with Bigrams:
models.logRegression(Bigram_Training, train_sentiments, Bigram_Testing, test_sentiments, 'Bigram')

####### Logistic Regssion Model with Trigrams:
models.logRegression(Trigram_Training, train_sentiments, Trigram_Testing, test_sentiments, 'Trigram')

###################### MULTINOMINAL NAIVE BAYES ######################

####### MULTINOMINAL Naive Bayes Model with DIRTY Data:
models.MultiNaiveBayes(Dirty_BOW_Training, train_sentiments, Dirty_BOW_Testing, test_sentiments, 'BOW Dirty')

####### MULTINOMINAL Naive Bayes Model with BOW's:
models.MultiNaiveBayes(BOW_Training, train_sentiments, BOW_Testing, test_sentiments, 'BOW')


############# OPTIMISING HYPERPARAMETERS FOR LOGISTIC REGRESSION ###############

# Optimising hyper parameters for Logistic Regression Model:
model = LogisticRegression()

# Defining 'parameter grid':
parameters = {
    'penalty' : ['l1','l2'],  # Regularization of the data (not all solvers support this)
    'C'       : [100, 10, 1.0, 0.1, 0.01],
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}
# Defining the Grid Search function with the desired parameters:
clf = GridSearchCV(model, param_grid = parameters, scoring = 'accuracy', cv = 10)
clf.fit(Vect_Training, train_sentiments)

# Printing the best score and parameters:
print("Best: %f using %s" % (clf.best_score_, clf.best_params_))

# Testing new one:
new_log = LogisticRegression(C = 1.0, penalty = 'l2', solver = 'newton-cg')
new_log.fit(Vect_Training, train_sentiments)
prediction = new_log.predict(Vect_Testing)
models.generate_metrics(test_sentiments, prediction, 'Logistic Regression with HyperParameters', 'TF-IDF')

########### OPTIMISING HYPERPARAMETERS FOR MULTINOMINAL NAIVE BAYES ############

# Optimising hyper parameters for Logistic Regression Model:
model = MultinomialNB()

# Defining 'parameter grid' for MultinominalNB:
params_NB = {'alpha': np.logspace(0,-9, num=100)}
# High alpha -> Underfitting
# Low alpha -> Overfitting

# Defining the Grid Search function with the desired parameters:
clf_nb = GridSearchCV(model, param_grid = params_NB, scoring = 'accuracy', cv = 10)

clf_nb.fit(Vect_Training, train_sentiments)

# Printing the best score and parameters:
print("Best: %f using %s" % (clf_nb.best_score_, clf_nb.best_params_))

# Testing new one:
new_NB = MultinomialNB(alpha=1.0)
new_NB.fit(Vect_Training, train_sentiments)
prediction = new_NB.predict(Vect_Testing)
models.generate_metrics(test_sentiments, prediction, 'Multinominal Naive Bayes with HyperParameters', 'TF-IDF')

################## TESTING TUNED ALGORITHMS ON TESTING DATA ####################


'''Need to use the testing data here and output it's efficiency'''

########### OLD STUFF IGNORE ME :D ############


# for name, sklearn_classifier in classifiers.items():
#      classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
#      classifier.train(features[:train_count])
#      accuracy = nltk.classify.accuracy(classifier, features[train_count:])
    #  print(F"{accuracy:.2%} - {name}")

# classifiers = {
#     "BernoulliNB": BernoulliNB(),
#     "ComplementNB": ComplementNB(),
#     "MultinomialNB": MultinomialNB(),
#     "KNeighborsClassifier": KNeighborsClassifier(),
#     "DecisionTreeClassifier": DecisionTreeClassifier(),
#     "RandomForestClassifier": RandomForestClassifier(),
#     "LogisticRegression": LogisticRegression(),
#     "MLPClassifier": MLPClassifier(max_iter=1000),
#     "AdaBoostClassifier": AdaBoostClassifier(),
# }

# FeatEng_Training, FeatEng_Testing = '', ''

# model_names = {
#     'Logistic Regression': models.logRegression(FeatEng_Training, train_sentiments, FeatEng_Testing, test_sentiments, '{}'),
#     'Multinominal Naive Bayes': models.MultiNaiveBayes(FeatEng_Training, train_sentiments, FeatEng_Testing, test_sentiments, '{}')
# }

# methods = {
#     '':"",
# }

# for feature_engineering_method, function in methods.items():
    # FeatEng_Training, FeatEng_Testing =  common.BagOfWords(train_reviews_original, test_reviews_original)
#     for name, sklearn_classifier in model_names.items():
#         pass
