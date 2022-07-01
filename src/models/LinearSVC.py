from .metrics import generate_metrics

from sklearn.svm import LinearSVC


def LinSVC(TrainingFeatures, TrainingSentiment, TestingFeatures,
           TestingSentiment, FeatEngName: str, **hyperParameters):
    '''
    Creates a LinearSVC Model using sklearn.svm. Outputs metrics and confusion
    matrix of the model generated.

    INPUTS:
    * TrainingFeatures - This is the 'feature engineered' training data to be
        used to train the model.
    * TrainingSentiment - This is the matching 'sentiment' values for the
        'feature engineered' training data (TrainingFeatures).
    * TestingFeatures - This is the 'feature engineered' testing data that you
        want the model to predict based upon (i.e. the reviews to predict the
        sentiment)
    * TestingSentiment - This is the matching 'sentiment' values for the
        'feature engineered' training data (TestingFeatures).
    * FeatEngName (str) - Name of the Feature Engineering used with the model
    ** hyperParameters (OPTIONAL ARGUMENT) - Allows the model to be configured
    with defined hyperparameters

    Outputs:
    * Terminal Output of metrics & Pyplot of Confusion Matrix
    '''
    lin = LinearSVC(**hyperParameters)
    lin_trained = lin.fit(TrainingFeatures, TrainingSentiment)

    # Checking that model has been trained correctly:
    print(lin_trained)

    # Making predictions with model:
    prediction = lin.predict(TestingFeatures)
    print(prediction)

    # Outputting measurement metrics for model:
    if len(hyperParameters) == 0:
        generate_metrics(TestingSentiment, prediction, 'Linear SVC',
                         FeatEngName)
    else:
        generate_metrics(TestingSentiment, prediction, 'Linear SVC with '
                         'Hyperparameters', FeatEngName)
