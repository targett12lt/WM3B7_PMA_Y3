from sklearn.linear_model import LogisticRegression
from .metrics import generate_metrics

def logRegression(TrainingData_X, TrainingData_Y, TestingData_X, TestingData_Y,
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
    lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
    lr_trained = lr.fit(TrainingData_X, TrainingData_Y)

    # Checking that model has been trained correctly:
    print(lr_trained)
    
    # Making predictions with model:
    prediction = lr.predict(TestingData_X)
    print(prediction)

    # Outputting measurement metrics for model:
    generate_metrics(TestingData_Y, prediction, 'Logistic Regression', FeatEngName)

   