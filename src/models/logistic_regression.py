from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from .metrics import ConfusionMatrix

def logRegression(TrainingData_X, TrainingData_Y, TestingData_X, TestingData_Y,
                  FeatEngName):
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
    print('Accuracy Score for Logistic Regression when used with ' + FeatEngName + ': ',
          accuracy_score(TestingData_Y, prediction))
    print('Classification report for Logisitic Regression model when used with ' + FeatEngName + ': \n'
          + classification_report(TestingData_Y, prediction, target_names=['Positive','Negative']))

    # Generating a confusion matrix:
    ConfusionMatrix(TestingData_Y, prediction)  # Generates a confusion matrix that is shown to the user

    # logisticregression = LogisticRegression()
    # logisticregression.fit(X_Train, Y_Train)
    # prediction = logisticregression.predict(X_Test)
    # print('Accuracy of Logisitc Regression: ', accuracy_score(prediction, Y_Test))
    # print('The classification report is as follows: \n'+classification_report(prediction, Y_Test))



   