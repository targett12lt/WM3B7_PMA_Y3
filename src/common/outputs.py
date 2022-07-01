from os.path import exists
from os import remove
import datetime

import csv


def CheckFileExists():
    '''Checks if CSV Output file already exists, if it does, it gets deleted
    and a new file is generated to output data to.'''
    # Check if CSV Output already exists:
    file_exists = exists('Outputs\ClassificationReports.csv')

    if file_exists:  # Removing old file to avoid confusion:
        remove('Outputs\ClassificationReports.csv')
    
    # Creating new file:
    text = ('Classification Report generated at: ' +
            str(datetime.datetime.now()))

    with open('Outputs\ClassificationReports.csv', 'a', newline='') as file:
        file.write(text)
        file.write('\n')


def WriteToCSV(classificationReport, algorithmName: str,
               FeatureEngineerName: str):
    ''' Appends the Classification report to the CSV file with additional
    information about the algorithm used and the feature engineering used to
    get the result. This function has no direct output (apart from adding to
    the CSV file)!

    INPUTS:
    * classificationReport - Classification report to be added to the CSV file
    * algorithmName (str) - Name of the algorithm/model used
    * FeatureEngineerName (str) - Name of the Feature Engineering used
    '''
    with open('Outputs\ClassificationReports.csv', 'a', newline='') as file:
        # writer = csv.writer(file)
        text = ('\nClassification Report for ' + algorithmName + ' using ' +
                FeatureEngineerName + '\n')
        file.write(text)
        file.write(classificationReport)
