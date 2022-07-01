import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Sklearn imports:
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import common


def generate_metrics(TestingData_Y, Predictions, ModelName: str,
                     FeatEngName: str):
    '''
    Generates metrics and outputs visual representation of Confusion Matrix
    using MatPlotLib Library.

    INPUTS:
    * TestingData_Y - Sentiments from Testing Data
    * Predictions - Matching predicted sentiments for same 'Testing Data'
    * ModelName (str) - Name of model used to make predictions
    * FeatEngName(str) - Name of feature engineering methods used to generate
        metrics.

    OUTPUTS:
    * Terminal Output of metrics & Pyplot of Confusion Matrix
    '''
    acc_score(TestingData_Y, Predictions, ModelName, FeatEngName)
    cr = ClassificationReport(TestingData_Y, Predictions, ModelName,
                              FeatEngName)
    common.ConfusionMatrix(TestingData_Y, Predictions, ModelName, FeatEngName)
    common.WriteToCSV(cr, ModelName, FeatEngName)


def acc_score(TestingData_Y, Predictions, ModelName: str, FeatEngName: str):
    '''
    Generates an accuracy score using SKLearn's "accuracy_score" method
    using the supplied "Y_Test" and "Prediction".

    INPUTS:
    * TestingData_Y - Sentiments from Testing Data
    * Predictions - Sentiments from Testing Data
    * ModelName (str) - Name of model used to make predictions
    * FeatEngName(str) - Name of feature engineering methods used to generate
        metrics.

    OUTPUTS:
    * AccScoreMetric - Decimal value showing accuracy of model
    '''
    AccScoreMetric = accuracy_score(TestingData_Y, Predictions)
    print('Accuracy Score for ' + ModelName + ' when used with ' +
          FeatEngName + ': ', AccScoreMetric)

    return AccScoreMetric


def ClassificationReport(TestingData_Y, Predictions, ModelName: str,
                         FeatEngName: str):
    '''
    Generates a classification report using SKLearn's "classification_report"
    method using the supplied "Y_Test" and "Prediction".

    INPUTS:
    * TestingData_Y - Sentiments from Testing Data
    * Predictions - Sentiments from Testing Data
    * ModelName (str) - Name of model used to make predictions
    * FeatEngName(str) - Name of feature engineering methods used to generate
        metrics.

    OUTPUTS:
    * ClassificationReportMetrics - Classification Report
    '''
    ClassificationReportMetrics = classification_report(TestingData_Y, Predictions, target_names=['Positive', 'Negative'])

    print('Classification report for ' + ModelName + ' model when used with ' +
          FeatEngName + ': \n' + ClassificationReportMetrics)

    return ClassificationReportMetrics
