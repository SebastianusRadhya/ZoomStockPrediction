# Zoom Stock Prediction
## Zoom Stock Prediction using Machine Learning in Python
This Zoom Stock Prediction program is a Machine Learning program to predict Zoom's stock closing price based on the opening price in the last three days. In making the prediction, this program utilizes Long Short Term Memory or LSTM. The development of this program aims to determine whether the closing price of shares can be predicted with the opening price of the previous three days using Machine Learning. In addition to predicting the closing price, in this program we can also do various other things. We can examine the Zoom stock dataset used in this program and also perform some operations on the dataset such as viewing the features' correlation in the dataset and the histogram for each feature. In addition, we can also perform a train test split in the program by specifying the test size and random state to generate evaluations based on several metrics. The GUI of this program was developped using tkinter.

## Running the Program
First of all, make sure you have installed all of the necessary libraries. I will provide a list of all the library's name used at the very bottom of this README file. To check which libraries are missing, run the program's main file "GUI.py" with a python compiler such as IDLE or Visual Studio Code and see the missing libraries from the error message. To install the missing libraries, do the command "pip install" with the library's name in the terminal/command prompt. If there's still an error with the library you have installed, please check if you have used the correct library name by searching on google. 

To run the program:
1. Run the program's main file "GUI.py". This will open the program's main page which is the Prediction Page.

![PredictionPage](https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/ss19.png?raw=true)

2. The program will have two main pages namely the Prediction Page and the Dataset Page.

Prediction Page:
1. To make a prediction, fill the three text box with the stock's opening price in the last three days.

![PredictionPageBox](https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/ss1.png?raw=true)

2. After all three text boxes are filled, click the Predict button which will generate the prediction in the text box below it.
3. To clear all text boxes, click on the Reset button.

Dataset Page:
1. To navigate into the Dataset Page, click on the Dataset navigation button at the top of the program.

![DatasetPage](https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/ss20.png?raw=true)

2. To plot a histogram for a feature, select a feature from the "Select Columns" dropdown menu and click the Plot Histogram button which will open up a new window with the histogram of the selected feature.

![DatasetPageHistogram](https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/ss10.png?raw=true)

![HistogramWindow](https://github.com/SebastianusRadhya/ZoomStockPrediction/blob/main/ss21.png?raw=true)
