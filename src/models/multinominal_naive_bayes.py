from sklearn.naive_bayes import MultinomialNB # Suitable for classification with discrete features, can work with td-idf but made for integer counts ideally

def MultiNaiveBayes(TrainingData_X, TrainingData_Y, TestingData_X):
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
    

