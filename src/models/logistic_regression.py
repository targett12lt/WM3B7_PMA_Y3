from .metrics import generate_metrics

from sklearn.linear_model import LogisticRegression

def logRegression(TrainingFeatures, TrainingSentiment, TestingFeatures, TestingSentiment,
                  FeatEngName: str, **hyperParameters):
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
  lr=LogisticRegression(**hyperParameters)
  lr_trained = lr.fit(TrainingFeatures, TrainingSentiment)

  # Checking that model has been trained correctly:
  print(lr_trained)
  
  # Making predictions with model:
  prediction = lr.predict(TestingFeatures)
  print(prediction)

  # Outputting measurement metrics for model:
  if len(hyperParameters) == 0:
    generate_metrics(TestingSentiment, prediction, 'Logistic Regression', FeatEngName)
  else:
    generate_metrics(TestingSentiment, prediction, 'Logistic Regression with Hyperparameters', FeatEngName)

   