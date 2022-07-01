import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from os import environ
from sklearn.metrics import confusion_matrix

'''
Ignores the QT Warnings when showing graphs using MatPlotLib or Seaborn
graphs on local machine

INSPIRATION: https://stackoverflow.com/questions/58194247/warning-qt-device-pixel-ratio-is-deprecated

'''
environ["QT_DEVICE_PIXEL_RATIO"] = "0"
environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
environ["QT_SCREEN_SCALE_FACTORS"] = "1"
environ["QT_SCALE_FACTOR"] = "1"


def visualise_sentiment_type(dataframe):
    '''Visualises the sentiment types stored in the supplied dataframe
    
    INPUTS:
    * dataframe - The dataframe you want the sentiments visualising

    OUTPUTS:
    * Returns Matplotlib 'pyplot' visualising the sentiment type in a 'catplot'
    '''
    ax = sns.catplot(data = dataframe, x = 'Sentiment', kind = 'count', 
                     palette='husl')
    ax.set(xlabel = 'Review Sentiment', ylabel = 'Number of Reviews')
    ax.set_xticklabels(['Negative', 'Positive'])
    plt.show()
    plt.savefig('Outputs\SentimentType_Train.png')


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

    # Generating file name to save graph to:
    name_file = ('Outputs\ConfusionMatrix_' + ModelName + '_' + FeatEngName +
                 '.png')

    # Visualising Confusion Matrix:
    class_label = ["Negative", "Positive"]
    ConfusionMatrixDF = pd.DataFrame(ConfusionMatrix, index=class_label,
                                     columns=class_label)
    sns.heatmap(ConfusionMatrixDF, annot=True, fmt="d")
    plt.title("Confusion Matrix for " + ModelName + ': ' + FeatEngName)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.savefig(name_file)
