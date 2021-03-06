# Importing packages:
import common
import models

# Importing external libraries:
import numpy as np
import os
import warnings

# Sklearn Imports:
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Ignoring 'Estimator Fit Fail' warnings when using CVGrid on incorrectly
# configured configurations:
warnings.filterwarnings('ignore')

# Getting file path to data:
os.chdir('....')  # Changing the working directory from script directory
cwd = os.getcwd()
base_folder_data = os.path.join(cwd, r'data\aclImdb')  # Specifying base dir

# Removing any old ClassificationReports:
common.CheckFileExists()

# Setting folders for training and testing data:
train_data = os.path.join(base_folder_data, 'train')
test_data = os.path.join(base_folder_data, 'test')

################################ IMPORTING DATA ################################
# Logic to choose data import method:
import_data = input('\nWould you like to import the data from the data folder'
                    ' (1) or the previously imported ".pkl" (2)? Use Option "1'
                    '" for marking purposes and use Option "2" for development'
                    ' purposes (loads slightly quicker as no need to iterate '
                    'through all review files). \n\nPlease enter the number '
                    'of your preferred option followed by "Enter": ')

# If running from scratch and creating PKL:
if '1' in import_data:
    # Training Data:
    print('Importing data from "data" folder...')
    training_data = common.importData(train_data)
    common.save_to_pkl(training_data, 'TrainingData.pkl')  # Saving data to pkl so it can be opened quickly

    # Test Data:
    test_data = common.importData(test_data)
    common.save_to_pkl(test_data, 'TestData.pkl')  # Saving data to pkl so it can be opened quickly

else:
    # Importing data from PKL file to make debugging/development quicker:
    print('Importing data from ".pkl" files...')
    training_data = common.read_from_pkl('TrainingData.pkl')
    test_data = common.read_from_pkl('TestData.pkl')
    
print('Import Complete!')

################################ CLEANING DATA #################################

common.add_cleaned_column(training_data, 'Review', 'CleanedReview')
common.add_cleaned_column(test_data, 'Review', 'CleanedReview')

print('Import complete. Extract from Pandas DF with cleaned column:\n',
      training_data, '\n')

############################### DATA EXPLORATION ###############################

print('NOTE: Please be aware that the script generates "Pylots" to visualise '
'confusion matrices and other graphs. When these are visualised the script '
'will PAUSE running until you CLOSE the graph! \n')

print('Visualising sentiment type on graph for "Training Data"... ')

# Visualising the number of each type of sentiment in the training set:
common.visualise_sentiment_type(training_data)

##################### SPLITTING DATA INTO TRAIN & VALIDATE #####################

print('Splitting training data into training and validation datasets...')

X_train, X_validate, Y_train, Y_validate = train_test_split(training_data.CleanedReview,
                                                            training_data.Sentiment,
                                                            test_size=0.15,
                                                            random_state=42)

# Ensuring Data types are in the correct format: 
Y_train = Y_train.astype('int')
Y_validate = Y_validate.astype('int')

# Defining reviews that haven't been cleaned:
X_Dirty_train, X_Dirty_validate, Y_Dirty_train, Y_Dirty_validate = train_test_split(training_data.Review, training_data.Sentiment, test_size=0.15, random_state=42)

# Fixing Data Types:
Y_Dirty_train = Y_train.astype('int')
Y_Dirty_validate = Y_validate.astype('int')

############################## FEATURE ENGINEERING #############################

print('Creating "Feature Engineering" features for training and '
      'validation dataset...')

# BAG OF WORDS (Unigram) with no cleaning:
Dirty_BOW_Training, Dirty_BOW_Validate =  common.BagOfWords(X_Dirty_train, X_Dirty_validate)

# BAG OF WORDS (Unigram):
BOW_Training, BOW_Validate = common.BagOfWords(X_train, X_validate)

# TF-IDF:
Vect_Training, Vect_Validate = common.tf_idf(X_train, X_validate)

# Bigram:
Bigram_Training, Bigram_Validate = common.n_gram(2, X_train, X_validate)

# Trigram:
Trigram_Training, Trigram_Validate = common.n_gram(3, X_train, X_validate)

######################## USING FEATURES TO TRAIN MODELS ########################

######################## LOGISTIC REGRESSION ########################

print('Testing Logistic Regression model with Feature Engineering approaches'
      '... This will generate "pyplot" graphs to render!')

####### LR With BOW With DIRTY Data:
models.logRegression(Dirty_BOW_Training, Y_Dirty_train, Dirty_BOW_Validate, Y_Dirty_validate, 'BOW Dirty')

####### Logistic Regssion Model with BOW's:
models.logRegression(BOW_Training, Y_train, BOW_Validate, Y_validate, 'BOW')

####### Logistic Regssion Model with TF-IDF:
models.logRegression(Vect_Training, Y_train, Vect_Validate, Y_validate, 'TF-IDF')

####### Logistic Regssion Model with Bigrams:
models.logRegression(Bigram_Training, Y_train, Bigram_Validate, Y_validate, 'Bigram')

####### Logistic Regssion Model with Trigrams:
models.logRegression(Trigram_Training, Y_train, Trigram_Validate, Y_validate, 'Trigram')

##################### MULTINOMINAL NAIVE BAYES ######################

print('Testing Multinominal Naive Bayes model with Feature Engineering '
      'approaches... This will generate "pyplot" graphs to render!')

###### Multinominal Naive Bayes Model with DIRTY Data:
models.MultiNaiveBayes(Dirty_BOW_Training, Y_Dirty_train, Dirty_BOW_Validate, Y_Dirty_validate, 'BOW Dirty')

####### Multinominal Naive Bayes Model with BOW's:
models.MultiNaiveBayes(BOW_Training, Y_train, BOW_Validate, Y_validate, 'BOW')

####### Multinominal Naive Bayes Model with TF-IDF:
models.MultiNaiveBayes(Vect_Training, Y_train, Vect_Validate, Y_validate, 'TF-IDF')

####### Multinominal Naive Bayes Model with Bigrams:
models.MultiNaiveBayes(Bigram_Training, Y_train, Bigram_Validate, Y_validate, 'Bigram')

####### Multinominal Naive Bayes Model with Trigrams:
models.MultiNaiveBayes(Trigram_Training, Y_train, Trigram_Validate, Y_validate, 'Trigram')

############################ LINEAR SVC #############################

print('Testing Linear SVC model with Feature Engineering approaches'
      '... This will generate "pyplot" graphs to render!')

###### Linear SVC Model with DIRTY Data:
models.LinSVC(Dirty_BOW_Training, Y_Dirty_train, Dirty_BOW_Validate, Y_Dirty_validate, 'BOW Dirty')

####### Linear SVC with Bag of Words:
models.LinSVC(BOW_Training, Y_train, BOW_Validate, Y_validate, 'BOW')

####### Linear SVC Model with TF-IDF:
models.LinSVC(Vect_Training, Y_train, Vect_Validate, Y_validate, 'TF-IDF')

####### Linear SVC Model with Bigrams:
models.LinSVC(Bigram_Training, Y_train, Bigram_Validate, Y_validate, 'Bigram')

####### Linear SVC Model with Trigrams:
models.LinSVC(Trigram_Training, Y_train, Trigram_Validate, Y_validate, 'Trigram')

############ OPTIMISING HYPERPARAMETERS FOR LOGISTIC REGRESSION ###############

print('Optimising Logistic Regression model by exhaustively searching for '
      'hyperparameters...\n')

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
models.logRegression(Vect_Training, Y_train, Vect_Validate, Y_validate,
                     'TF-IDF', **clf.best_params_)

########### OPTIMISING HYPERPARAMETERS FOR MULTINOMINAL NAIVE BAYES ############

print('Optimising Multinominal Naive Bayes model by exhaustively searching for'
      ' hyperparameters...\n')

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
models.MultiNaiveBayes(Bigram_Training, Y_train, Bigram_Validate, Y_validate,
                       'Bigram', **clf_nb.best_params_)

################## OPTIMISING HYPERPARAMETERS FOR LINEAR SVC ###################

print('Optimising Linear SVC model by exhaustively searching for '
      'hyperparameters...\n')

# Creating Linear SVC:
model_svc = LinearSVC()

# Defining 'parameter grid' for Linear SVC:
parameters_linear_svc = {
    'penalty' : ['l1','l2'],  # Regularization of the data (not all solvers support this)
    'loss': ['hinge', 'squared_hinge'],
    'C'       : [100, 10, 1.0, 0.1, 0.01],
                        }

# Defining the Grid Search function with the desired parameters:
clf_svc = GridSearchCV(model_svc, param_grid = parameters_linear_svc,
                       scoring = 'accuracy', cv = 10)

clf_svc.fit(Vect_Training, Y_train)

# Printing the best score and parameters:
print("Best score achieved using %s" % (clf_svc.best_params_))

# Testing new one:
models.LinSVC(Vect_Training, Y_train, Vect_Validate, Y_validate,
              'TF-IDF', **clf_svc.best_params_)

################## TESTING TUNED ALGORITHMS ON TESTING DATA ####################

print('Testing optimised feature engineering-model (with optimised'
      ' hyperparameter) combinations to test performance on "Test" data')

#### Setting variables for test data:
# Reviews:
test_reviews = test_data.CleanedReview

# Sentiments:
test_sentiments = test_data.Sentiment
test_sentiments = test_sentiments.astype('int')

#### Completing Feature engineering on the test data:

print('Generating Engineered Features for training data...')

# TF-IDF:
Vect_Training, Vect_Testing = common.tf_idf(X_train, test_reviews)

# Bigram:
Bigram_Training, Bigram_Testing = common.n_gram(2, X_train, test_reviews)

# Measuring model on test dataset using the three models with optimised 
# feature engineering and hyperparameters:

# Logistic Regression with optimal feature engineering & hyperparameters:
models.logRegression(Vect_Training, Y_train, Vect_Testing, test_sentiments,
                     'TF-IDF (Test Data)', **clf.best_params_)

# Multinominal Naive Bayes with optimal feature engineering & hyperparameters:
models.MultiNaiveBayes(Bigram_Training, Y_train, Bigram_Testing,
                       test_sentiments, 'Bigram (Test Data)',
                       **clf_nb.best_params_)

# Linear SVC with optimal feature engineering & hyperparameters:
models.LinSVC(Vect_Training, Y_train, Vect_Testing, test_sentiments,
              'TF-IDF (Test Data)', **clf_svc.best_params_)

print('All calculations are complete. \n\nClassification Reports can be found'
      ' in "outputs\ClassificationReports.csv"\n\nConfusion matrices can be '
      'found with the following file structure '
      '"outputs\ConfusionMatrix_{NameOfModel}_{FeatureEngineeringName}.png"')
