from .metrics import generate_metrics

from sklearn.svm import LinearSVC

'''SVM and SVC are different implementations of the same algorithm, LinearSVC is based on liblinear and ONLY supports a linear kernel. 
Because the implementations are different in practice you will get different results, the most important ones being that LinearSVC only 
supports a linear kernel, is faster and can scale a lot better. HAS HIGHER PERFORMANCE THAT '''

def LinSVC(TrainingFeatures, TrainingSentiment, TestingFeatures, TestingSentiment,
           FeatEngName: str, **hyperParameters):
    '''

    Information here

    Inputs: 
    * TrainingFeatures - This is the 'feature engineered' training data to be used
        to train the model.
    * TrainingSentiment - This is the matching 'sentiment' values for the 'feature 
        engineered' training data (TrainingData_X).
    * TestingFeatures - This is the 'feature engineered' testing data that you
        want the model to predict based upon (i.e. the reviews to predict the 
        sentiment)
    * TestingSentiment - 
    * FeatEngName -
    ** hyperParameters - 

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
        generate_metrics(TestingSentiment, prediction, 'Linear SVC', FeatEngName)
    else:
        generate_metrics(TestingSentiment, prediction, 'Linear SVC with Hyperparameters', FeatEngName)

