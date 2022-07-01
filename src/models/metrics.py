import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Sklearn imports:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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
    ClassificationReport(TestingData_Y, Predictions, ModelName, FeatEngName)
    ConfusionMatrix(TestingData_Y, Predictions, ModelName, FeatEngName)


def ConfusionMatrix(TestingData_Y, Predictions, ModelName: str,
                    FeatEngName: str):
    '''
    Generates a confusion matrix using SKLearn's "Confusion Matrix" method
    using the supplied "Y_Test" and "Prediction".

    INPUTS:
    * TestingData_Y - Sentiments from Testing Data
    * Predictions - Sentiments from Testing Data
    * ModelName (str) - Name of model used to make predictions
    * FeatEngName(str) - Name of feature engineering methods used to generate
        metrics.

    OUTPUTS:
    * Pyplot of Confusion Matrix
    '''
    # Creating Confusion Matrix:
    ConfusionMatrix = confusion_matrix(TestingData_Y, Predictions,
                                       labels=[1, 0])

    # Visualising Confusion Matrix:
    class_label = ["Negative", "Positive"]
    ConfusionMatrixDF = pd.DataFrame(ConfusionMatrix, index=class_label,
                                     columns=class_label)
    sns.heatmap(ConfusionMatrixDF, annot=True, fmt="d")
    plt.title("Confusion Matrix for " + ModelName + ': ' + FeatEngName)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


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
