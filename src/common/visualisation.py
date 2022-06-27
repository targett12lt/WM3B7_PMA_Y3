import seaborn as sns
import matplotlib.pyplot as plt

from os import environ

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
    ax = sns.catplot(data = dataframe, x = 'PositiveReview', kind = 'count', 
                     palette='husl')
    ax.set(xlabel = 'Review Sentiment', ylabel = 'Number of Reviews')
    ax.set_xticklabels(['Negative', 'Positive'])
    plt.show()





