from sklearn.naive_bayes import MultinomialNB # Suitable for classification with discrete features, can work with td-idf but made for integer counts ideally
from .metrics import generate_metrics

def MultiNaiveBayes(TrainingFeatures, TrainingSentiment, TestingFeatures, TestingSentiment,
                    FeatEngName: str):
    '''
    Information here
    
    Inputs: 
    * TrainingData_X - This is the 'feature engineered' training data to be used
      to train the model.
    * TrainingData_Y - This is the matching 'sentiment' values for the 'feature 
      engineered' training data (TrainingData_X).
    * TestingData_X - This is the 'feature engineered' testing data that you
      want the model to predict based upon (i.e. the reviews to predict the 
      sentiment)

    Outputs: 
    *

    '''
    nb = MultinomialNB()
    nb_trained = nb.fit(TrainingFeatures, TrainingSentiment)

    # Checking that model has been trained correctly:
    print(nb_trained)

    # Making Predictions with model:
    prediction = nb.predict(TestingFeatures)
    print(prediction)

    # Outputting measurement metrics for model:
    generate_metrics(TestingSentiment, prediction, 'Multinominal Naive Bayes', FeatEngName)

