from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ConfusionMatrix(Y_Test, Prediction):
    '''Generates a confusion matrix using SKLearn's "Confusion Matrix" method
    using the supplied "Y_Test" and "Prediction".'''
    # Creating Confusion Matrix:
    ConfusionMatrix = confusion_matrix(Y_Test, Prediction, labels=[1, 0])
    # fig, ax = plot_confusion_matrix(ConfusionMatrix, figsize = (10,10))
    # plt.show()

    # Visualising Confusion Matrix:
    class_label = ["Negative", "Positive"]
    ConfusionMatrixDF = pd.DataFrame(ConfusionMatrix, index = class_label, columns = class_label)
    sns.heatmap(ConfusionMatrixDF, annot = True, fmt = "d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def accuracy_score():
    '''
    Info here

    INPUTS:
    *
    
    OUTPUTS:
    *
    '''


