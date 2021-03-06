from sklearn.naive_bayes import MultinomialNB
from .metrics import generate_metrics


def MultiNaiveBayes(TrainingFeatures, TrainingSentiment, TestingFeatures,
                    TestingSentiment, FeatEngName: str, **hyperParameters):
    '''
    Creates a Multinomonal Naive Bayes Model using sklearn.svm. Outputs
    metrics and confusion matrix of the model generated.

    Inputs:
    * TrainingData_X - This is the 'feature engineered' training data to be
      used to train the model.
    * TrainingData_Y - This is the matching 'sentiment' values for the 'feature
      engineered' training data (TrainingData_X).
    * TestingData_X - This is the 'feature engineered' testing data that you
      want the model to predict based upon (i.e. the reviews to predict the
      sentiment)

    Outputs:
    * Terminal Output of metrics & Pyplot of Confusion Matrix

    '''
    nb = MultinomialNB(**hyperParameters)
    nb_trained = nb.fit(TrainingFeatures, TrainingSentiment)

    # Checking that model has been trained correctly:
    # print(nb_trained)  # FOR DEBUGGING PURPOSES

    # Making Predictions with model:
    prediction = nb.predict(TestingFeatures)
    # print(prediction)  # FOR DEBUGGING PURPOSES

    # Outputting measurement metrics for model:
    if len(hyperParameters) == 0:
        generate_metrics(TestingSentiment, prediction, 'Multinominal Naive '
                         'Bayes', FeatEngName)
    else:
        generate_metrics(TestingSentiment, prediction, 'Multinominal Naive '
                         'Bayes with Hyperparameters', FeatEngName)
