# WM3B7_PMA_Y3

## About The Project
This code aims to implement three suitable machine learning algorithms and optimise them using a combination of hyperparameters and suitable feature engineering features. Their performance will be measured using suitable machine learning performance metrics. The machine learning algorithms will be used to perform sentiment analysis on movie reviews from the IMBD website.  

Outputs are stored in the 'outputs' folder - this includes the 'ClassificationReports.csv' which contains all the classification reports
generated when creating the initial models, in addition to the optimised models with hyperparameters. This folder also includes all the 
confusion matrices generated. These follow the following naming convention: "ConfusionMatrix_{NameOfModel}_{FeatureEngineeringName}.png".  

## Built With:
* Anaconda Python Environment 'WM3B7'
* datetime
* os
* sklearn
* nltk
* pandas
* re
* matplotlib
* seaborn
* numpy
* warnings


## Getting Started
Set IDE Working directory to parent folder (THIS FOLDER) which contains 'ReadMe.md'

Run 'SA_1943174.py' (in 'src' folder) script using the Anaconda 'WM3B7' environment. The script will automatically 
download the required NLTK files if they aren't already on your local PC. 

FOR MARKING PURPOSES: Please ensure that you choose option '1' when prompted (followed by enter) - this ensures the 
data is imported from the file structure, rather than the '.pkl' files.

NOTE: Please be aware that the script generates "Pylots" to visualise confusion matrices and other graphs. 
When these are visualised the script will pause running until you close the graph!  

### Prerequisites:
* PC has Anaconda 'WM3B7' environment installed and activated.

### Installation
1.  Clone the repository on to local machine using:  
`git clone https://github.com/targett12lt/WM3B7_PMA_Y3.git`
2. Launch Project Folder and Run in local IDE (e.g. Microsoft Visual Studio Code)  

## GIT
To download the latest release of this source code please visit:  
https://github.com/targett12lt/WM3B7_PMA_Y3
