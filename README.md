# Zoom Stock Prediction
## Zoom Stock Prediction using Machine Learning in Python
This Zoom Stock Prediction program is a Machine Learning program using Python to predict Zoom's stock closing price based on the opening price in the last three days. In making the prediction, this program utilizes Long Short Term Memory or LSTM. The development of this program aims to determine whether the closing price of shares can be predicted with the opening price of the previous three days using Machine Learning. In addition to predicting the closing price, in this program we can also do various other things. We can examine the Zoom stock dataset used in this program and also perform some operations on the dataset such as viewing the features' correlation in the dataset and the histogram for each feature. In addition, we can also perform a train test split in the program by specifying the test size and random state to generate evaluations based on several metrics. The GUI of this program was developped using tkinter.

## Running the Program
First of all, make sure you have installed all of the necessary libraries. I will provide a list of all the library's name used at the very bottom of this README file. To check which libraries are missing, run the program's main file "GUI.py" with a python compiler such as IDLE or Visual Studio Code and see the missing libraries from the error message. To install the missing libraries, do the command "pip install" with the library's name in the terminal/command prompt. If there's still an error with the library you have installed, please check if you have used the correct library name by searching on google. 

To run the program:
1. Run the program's main file "GUI.py". This will open the program's main page which is the Prediction Page.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss19.png?raw=true" width="600">

2. The program will have two main pages namely the Prediction Page and the Dataset Page.

Prediction Page:
1. To make a prediction, fill the three text box with the stock's opening price in the last three days.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss1.png?raw=true" width="600">

2. After all three text boxes are filled, click the Predict button which will generate the prediction in the text box below it.
3. To clear all text boxes, click on the Reset button.

Dataset Page:
To navigate into the Dataset Page, click on the Dataset navigation button at the top of the program.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss20.png?raw=true" width="600">

To plot a histogram for a feature:
Select a feature from the "Select Columns" dropdown menu and click the Plot Histogram button which will open up a new window with the histogram of the selected feature.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss10.png?raw=true" width="600">

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss21.png?raw=true" width="400">

To see the dataset's feature correlation: 
Click on the Show Dataset Correlations button which will open up a new window with a Heatmap of the dataset's feature correlation.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss12.png?raw=true" width="600">

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss22.png?raw=true" width="400">

To evaluate this program's algorithm:
1. Click on the Train Test Dataset button. This will open up the Train Test Operations window where you can evaluate the program's prediction algorithm with the option to manually input the train test split's test size and random state. To generate the evaluations, click on the Train Test Split button.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss13.png?raw=true" width="600"><br>
   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss23.png?raw=true" width="260">
   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss15.png?raw=true" width="260">
   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss17.png?raw=true" width="260">
   
   
2. To plot a scatter plot of the prediction values vs the actual values, click on the Plot Prediction vs Actual button which will open two windows with scatter plots of the prediction values vs the actual values in the train set and the test set.

   <img src="https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/screenshots/ss18.png?raw=true" width="260">

## Necesarry Python Libraries
- pandas
- numpy
- tkinter
- pandastable
- matplotlib
- keras
- scikit-learn
- seaborn
