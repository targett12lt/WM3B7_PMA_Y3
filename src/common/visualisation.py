import seaborn as sns
import matplotlib.pyplot as plt

from os import environ

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

