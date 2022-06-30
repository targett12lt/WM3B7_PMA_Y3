import seaborn as sns
import matplotlib.pyplot as plt

from os import environ

from sklearn.metrics import plot_roc_curve, roc_curve

# Space to visualise term frequency, etc.
# compare negative vs positive sentiment in graph for basic data exploration/analysis

# THIS DOESN'T WORK.... LOOK AT THIS LATER :/
def ignore_qt_warnings():
    '''
    Ignores the QT Warnings when showing graphs using MatPlotLib or Seaborn
    graphs on local machine

    INSPIRATION: https://stackoverflow.com/questions/58194247/warning-qt-device-pixel-ratio-is-deprecated
    
    '''
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ == '__main__':
    ignore_qt_warnings()

def visualise_sentiment_type(dataframe):
    '''Visualises the dataframe supplied and '''
    ax = sns.catplot(data = dataframe, x = 'Sentiment', kind = 'count', 
                     palette='husl')
    ax.set(xlabel = 'Review Sentiment', ylabel = 'Number of Reviews')
    # ax.set_xticklabels(['Negative', 'Positive'])
    plt.show()

def visualise_fd_histogram(positive_words, negative_words):
    '''Creates histogram sshowing frequency density and the most common words'''
    print(type(positive_words), ' LOOK HERE ', positive_words)
    plt.bar(positive_words.keys(), positive_words.values(), color='g')
    # plt.bar(negative_words.keys(), negative_words.values(), color='g')
    plt.show()

# def visualise
#     roc_curve
#     plot_roc_curve






