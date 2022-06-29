from statistics import mean
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import *
import common
import models
import nltk

nltk.download('vader_lexicon')  # required for sentiment intensity analyzer


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer


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

# Training Data:
# print('Importing Training Data to DataFrame...')
# training_data = common.importData(train_data)
# common.save_to_pkl(training_data, 'TrainingData.pkl')  # Saving data to pkl so it can be opened quickly

# Test Data:
# print('Importing Test Data to DataFrame...')
# test_data = common.importData(test_data)
# common.save_to_pkl(test_data, 'TestData.pkl')  # Saving data to pkl so it can be opened quickly

# Importing data from PKL file to make debugging/development quicker:
training_data = common.read_from_pkl('TrainingData.pkl')
test_data = common.read_from_pkl('TestData.pkl')

################################ CLEANING DATA #################################

common.add_cleaned_column(training_data, 'Review', 'CleanedReview')
common.add_cleaned_column(test_data, 'Review', 'CleanedReview')

print('Cleaned Pandas DF:')
print(training_data)

############################### DATA EXPLORATION ###############################

# NEED TO PUT SOME DATA EXPLORATION/VISUALISATION HERE
# common.visualise_sentiment_type(training_data)

############################## FEATURE ENGINEERING #############################

# bag_of_words = common.BagOfWords(training_data)

# common.FrequencyDistribution(training_data['CleanedReview'])

# TFIDF_Output = common.tf_idf(training_data)  # Returns X and Y Vals from Tfidf Vectorizer
# TFIDF_Output_Test = common.tf_idf(test_data)  # Returns X and Y Vals from Tfidf Vectorizer

# Creating a Document Term Matrix:
# DocTermMatrix = common.DocumentTermMatrix(training_data, TFIDF_Output[0], TFIDF_Output[1], TFIDF_Output[2])
# DocTermMatrixTest = common.DocumentTermMatrix(test_data, TFIDF_Output_Test[0], TFIDF_Output_Test[1], TFIDF_Output_Test[2])

####################### SPLITTING DATA INTO TEST & TRAIN #######################

# X_Train, Y_Train = DocTermMatrix.drop(['Sentiment'], axis = 1), DocTermMatrix['Sentiment']
# X_Test, Y_Test = DocTermMatrixTest.drop(['Sentiment'], axis = 1), DocTermMatrixTest['Sentiment']

# Testing using original data:
# X_Train, Y_Train = training_data.drop(['Sentiment'], axis = 1), training_data['Sentiment']
# X_Test, Y_Test = test_data.drop(['Sentiment'], axis = 1), test_data['Sentiment']

# X_train, X_validate, Y_train, Y_Validate = train_test_split(training_data.CleanedReview, training_data.Sentiment, test_size=0.2, random_state=42)

# X_validate = X_validate.astype('int')
# Y_Validate = Y_Validate.astype('int')
# NEW

# Setting variables:
train_reviews = training_data.CleanedReview
train_sentiments = training_data.Sentiment
train_sentiments = train_sentiments.astype('int')

test_reviews = test_data.CleanedReview
test_sentiments = test_data.Sentiment
test_sentiments = test_sentiments.astype('int')

# BAG OF WORDS:
BOW_Training, BOW_Testing = common.BagOfWords(training_data.CleanedReview, test_data.CleanedReview)

# TF-IDF:
Vect_Training, Vect_Testing, featureNames = common.tf_idf(training_data.CleanedReview, test_data.CleanedReview)

######################## USING FEATURES TO TRAIN MODELS ########################

####### Logistic Regssion Model with BOW's:

models.logRegression(BOW_Training, train_sentiments, BOW_Testing, test_sentiments, 'BOW')

####### Logistic Regssion Model with TF-IDF:
models.logRegression(Vect_Training, train_sentiments, Vect_Testing, test_sentiments, 'TF-IDF')

######## OLD CODE 

# predictions = models.linRegression(X_Train, Y_Train, X_Test, Y_Test)
# clf = Pipeline(steps =[
#     ('preprocessing', CountVectorizer()),
#     ('classifier', LogisticRegression(dual=False, max_iter=2000))
# ])

# # Fitting the model:
# clf.fit(X_Train, Y_Train)

# # Scoring them:
# # clf.score(X_valid, Y_valid)


# clf.score(X_Test,Y_Test)

# from nltk.sentiment import SentimentIntensityAnalyzer

# sia = SentimentIntensityAnalyzer()

# # Specifying classifiers:
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

# Getting positive and negative FD's:
# positive_fd = nltk.FreqDist(training_data[training_data['Sentiment'] == 'Positive'])
# negative_fd = nltk.FreqDist(training_data[training_data['Sentiment'] == 'Negative'])

# common_set = set(positive_fd).intersection(negative_fd)

# for word in common_set:
#     del positive_fd[word]
#     del negative_fd[word]

# Using Frequency Density to get 100 most positive and negative words:
# top_100_positive = {word for word, count in positive_fd.most_common(100)}
# top_100_negative = {word for word, count in negative_fd.most_common(100)}

# common.visualise_fd_histogram(top_100_positive, top_100_negative)

# def extract_features(text):
#     features = dict()
#     wordcount = 0
#     compound_scores = list()
#     positive_scores = list()

#     for sentence in nltk.sent_tokenize(text):
#         for word in nltk.word_tokenize(sentence):
#             if word.lower() in top_100_positive:
#                 wordcount += 1
#         compound_scores.append(sia.polarity_scores(sentence)["compound"])
#         positive_scores.append(sia.polarity_scores(sentence)["pos"])

#     # Adding 1 to the final compound score to always have positive numbers
#     # since some classifiers you'll use later don't work with negative numbers.
#     features["mean_compound"] = mean(compound_scores) + 1
#     features["mean_positive"] = mean(positive_scores)
#     features["wordcount"] = wordcount

#     return features

# features = [
#     (extract_features(training_data[training_data['Sentiment'] == 'Positive'])
#     for review in training_data[training_data['Sentiment'] == 'Positive'])
# ]
# features.extend([
#     (extract_features(training_data[training_data['Sentiment'] == 'Negative'])
#     for review in training_data[training_data['Sentiment'] == 'Negative'])
# ])

# # Looping through classifiers:
# train_count = len(features)

# for name, sklearn_classifier in classifiers.items():
#      classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
#      classifier.train(features[:train_count])
#      accuracy = nltk.classify.accuracy(classifier, features[train_count:])
#      print(F"{accuracy:.2%} - {name}")

