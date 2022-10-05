import pandas as pd
import numpy as np
from statistics import correlation
import tkinter as tk
from tkinter import LEFT, ttk
from tkinter.messagebox import NO
from turtle import bgcolor, left, right
import Zoom
import os
from pandastable import Table, TableModel,config
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from math import sqrt
from matplotlib.colors import ListedColormap

window = tk.Tk()
window.title("Zoom Stock Prediction")
window.geometry('800x400')

tabControl = ttk.Notebook(window)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)

tabControl.add(tab1, text='Prediction')
tabControl.add(tab2, text='Dataset')
tabControl.pack(expand = 1, fill ="both")
tab1.grid_columnconfigure(0, weight=3)
tab1.grid_columnconfigure(1, weight=1)
tab1.grid_columnconfigure(2, weight=1)
tab1.grid_columnconfigure(3, weight=1)
tab1.grid_columnconfigure(4, weight=1)
tab1.grid_columnconfigure(5, weight=3)

label = tk.Label(tab1, text="Zoom Stock Prediction", font=("Century Gothic Bold", 20))
label.grid(column=2,row=0,pady=20)

instructions = tk.Label(tab1, text="Input the opening prices in the last 3 days",font=("Century Gothic",11))
instructions.grid(column=2,row=1)

first = tk.Entry(tab1)
first.grid(column=1,row=2,pady=15)
second = tk.Entry(tab1)
second.grid(column=2,row=2,pady=15)
third = tk.Entry(tab1)
third.grid(column=3,row=2,pady=15)

def predict_function():
    results = Zoom.predict_price(float(first.get()),float(second.get()),float(third.get()))
    predicted.configure(state="normal")
    predicted.delete(0,tk.END)
    predicted.insert(tk.INSERT,results[0][0])
    predicted.configure(state="disabled", disabledforeground='black')

def clear_entry():
    first.delete(0,tk.END)
    second.delete(0,tk.END)
    third.delete(0,tk.END)
    predicted.configure(state="normal")
    predicted.delete(0,tk.END)
    predicted.configure(state="disabled", disabledforeground='black')

predict = tk.Button(tab1, text='Predict',width = 15, bg='#2D8CFF', fg='white', font=("Century Gothic Bold",11),command=predict_function)
predict.grid(column=2,row=3,pady=15,padx=(0,150))

reset = tk.Button(tab1, text="Reset",width=10,bg="red",fg='white',font=("Century Gothic Bold",11),command=clear_entry)
reset.grid(column=2,row=3,pady=15,padx=(150,0))

predictedlabel = tk.Label(tab1, text="Closing Price The Next Day: ",font=("Century Gothic",10))
predictedlabel.grid(column=1,row=4,pady=20)
predicted = tk.Entry(tab1,width=40)
predicted.configure(state="disabled")
predicted.grid(column=2,row=4,pady=20)

# Tab2
tab2.grid_columnconfigure(0, weight=2)
tab2.grid_columnconfigure(1, weight=1)
tab2.grid_columnconfigure(2, weight=1)
tab2.grid_columnconfigure(3, weight=1)
tab2.grid_columnconfigure(4, weight=1)
tab2.grid_columnconfigure(5, weight=2)

tab2.grid_rowconfigure(0, weight=11)
tab2.grid_rowconfigure(1, weight=1)
tab2.grid_rowconfigure(2, weight=1)
tab2.grid_rowconfigure(3, weight=1)
tab2.grid_rowconfigure(4, weight=1)
tab2.grid_rowconfigure(5, weight=4)

framefortable = tk.LabelFrame(tab2, text="Zoom Stock Dataset")
framefortable.place(height=200, width=750,relx=0.03,rely=0.025)

df = Zoom.dataset_train
df_table = Table(framefortable, dataframe=df)
df_table.show()
options = {'colheadercolor':'green','floatprecision': 5}
config.apply_options(options, df_table)
df_table.show()

def correlation_function():
    Zoom.corr()


def traintest_function():
    traintestwindow = tk.Toplevel(tab2,width=300,height=400)
    traintestwindow.minsize(300, 400)
    traintestwindow.resizable(300,400)
    label = tk.Label(traintestwindow,text="Train Test Operations", font=("Century Gothic Bold", 12))
    label.grid(row=0,column=0,pady=15,padx=10)
    labelframe = tk.LabelFrame(traintestwindow, text="Options")
    labelframe.grid(row=1,column=0,pady=20,padx=10,sticky="w")
    testsizelabel = tk.Label(labelframe,text="Test Size: ")
    testsizelabel.grid(row=0,column=0,sticky="w")
    testentry = tk.Entry(labelframe)
    testentry.insert(tk.INSERT,"0.25")
    testentry.grid(row=0,column=1)
    randomstatelabel = tk.Label(labelframe,text="Random State: ")
    randomstatelabel.grid(row=1,column=0,sticky="w")
    randomentry = tk.Entry(labelframe)
    randomentry.insert(tk.INSERT,"0")
    randomentry.grid(row=1,column=1)
    X_train, X_test, y_train, y_test = train_test_split(Zoom.X, Zoom.y, test_size = float(testentry.get()), shuffle=False, random_state=float(randomentry.get()))
    model = Sequential()
    model.add(Zoom.LSTM(50, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, verbose=0)
    y_pred = model.predict(X_test)
    def traintest_split():
        mar_result.config(state="normal")
        mar_result.insert(tk.INSERT,mean_absolute_percentage_error(y_test, y_pred))
        mar_result.config(state="disabled")
        
        mse_result.config(state="normal")
        mse_result.insert(tk.INSERT,mean_squared_error(y_test, model.predict(X_test), squared=False))
        mse_result.config(state="disabled")
        
        rmse_result.config(state="normal")
        rmse_result.insert(tk.INSERT,sqrt(mean_squared_error(y_test, model.predict(X_test), squared=False)))
        rmse_result.config(state="disabled")
        
        r2_result.config(state="normal")
        r2_result.insert(tk.INSERT,r2_score(y_test, y_pred))
        r2_result.config(state="disabled")

    def plotpredict():
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.title("Zoom Dataset Actual vs Predicted (Test Set)")

        fig, ax = plt.subplots()
        ax.scatter(y_train, model.predict(X_train))
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.title("Zoom Dataset Actual vs Predicted (Train Set)")
        plt.show()

    button = tk.Button(labelframe,text="Train Test Split",bg="#77DD77",width=20,command=traintest_split)
    button.grid(row=2,column=1,pady=(15,0))
    labelframe_eval = tk.LabelFrame(traintestwindow, text="Evaluation Metrics")
    labelframe_eval.grid(row=2,column=0,pady=20,padx=10,sticky="w")
    marlabel = tk.Label(labelframe_eval,text="Mean Absolute Percentage Error: ")
    marlabel.grid(row=0,column=0,sticky="w")
    mar_result = tk.Entry(labelframe_eval,disabledforeground='black')
    mar_result.config(state="disabled")
    mar_result.grid(row=0,column=1)
    mse_label = tk.Label(labelframe_eval,text="Mean Squared Error: ")
    mse_label.grid(row=1,column=0,sticky="w")
    mse_result = tk.Entry(labelframe_eval,disabledforeground='black')
    mse_result.config(state="disabled")
    mse_result.grid(row=1,column=1)
    rmse_label = tk.Label(labelframe_eval,text="Root Mean Squared Error: ")
    rmse_label.grid(row=2, column=0,sticky="w")
    rmse_result = tk.Entry(labelframe_eval,disabledforeground='black')
    rmse_result.config(state="disabled")
    rmse_result.grid(row=2,column=1)
    r2_label = tk.Label(labelframe_eval,text="R-Squared Score: ")
    r2_label.grid(row=3, column=0,sticky="w")
    r2_result = tk.Entry(labelframe_eval,disabledforeground='black')
    r2_result.config(state="disabled")
    r2_result.grid(row=3,column=1)
    predictplot_button = tk.Button(traintestwindow,text="Plot Prediction vs Actual",bg="#ADD8E6",width=30,command=plotpredict)
    predictplot_button.grid(row=4,column=0,pady=(25,0))
    
    
def plot_hist():
    if(combobox.current() > -1):
        plt.figure()
        plt.hist(df[combobox.get()].values)
        plt.title(combobox.get()+" Histogram")
        plt.show()
    else:
        tk.messagebox.showerror(title="Column Not Selected", message="Please select a column!")

correlation_button = tk.Button(tab2, text="Show Dataset Correlations", command=correlation_function)
correlation_button.grid(row=3,column=0)

traintest_button = tk.Button(tab2, text="Train Test Dataset", command=traintest_function, width=20)
traintest_button.grid(row=4,column=0)


histlabel = tk.LabelFrame(tab2, text="Histogram")
histlabel.place(height=100, width=375,relx=0.5,rely=0.6)
histlabel.grid_columnconfigure(0, weight=1)
histlabel.grid_columnconfigure(1, weight=1)
histlabel.grid_columnconfigure(2, weight=1)
histlabel.grid_columnconfigure(3, weight=1)

selectlabel = tk.Label(histlabel,text="Select Columns: ")
selectlabel.grid(row=0,column=0,pady=4,padx=1)

current_var = tk.StringVar()
options = ['Open','High','Low', 'Close', 'Adj Close', 'Volume']
combobox = ttk.Combobox(histlabel, textvariable=current_var, width=11,values=options)
combobox.grid(row=1, column=0,padx=1,pady=2)

printhist_button = tk.Button(histlabel, text="Plot Histogram",width=24,bg='#77DD77',command=plot_hist)
printhist_button.grid(column=2,row=1,pady=4,columnspan=2)

window.mainloop()