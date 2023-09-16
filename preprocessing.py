#cleaveland hear disease dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def get_dataset(name):
    if name == 'circles':
        data = pd.read_csv('union_of_convex_datasets/data_dist_circle.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "checkerboard":
        data = pd.read_csv('union_of_convex_datasets/data_dist_circle.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "churn":
        df = pd.read_csv("datasets/Churn_Modelling.xls")
        df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
        df['Geography'].replace(['France', 'Germany', 'Spain'], [0, 1, 2], inplace=True)
        df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
        df = df[(df != '?').all(axis=1)]
        df.sample(frac=1)
        data = df.to_numpy()
        data_x = (data[:, 0:-1] / np.amax(np.abs(data[:, 0:-1]), axis=0))
        data_y = data[:, -1]
        data_y = np.where(data_y == 0, 0, 1)
        # print('xxxxxxx')
        # print(data_x.shape)
        # print(np.amax(data_x[:, 10]))
        # print(np.amin(data_x[:, 10]))
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "telco_churn":
        df = pd.read_csv("datasets/TelcoCustomerChurn.xls")
        df.drop(['customerID'], axis=1, inplace=True)
        df = df[(df != '?').all(axis=1)]
        df = df[(df != ' ').all(axis=1)]
        df = df[(df != '').all(axis=1)]
        df['gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
        df['Partner'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['Dependents'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['PhoneService'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['MultipleLines'].replace(['No phone service', 'No', 'Yes'], [0, 1, 2], inplace=True)
        df['InternetService'].replace(['DSL', 'Fiber optic', 'No'], [0, 1, 2], inplace=True)
        df['OnlineSecurity'].replace(['No', 'Yes', 'No internet service'], [0, 1, 2], inplace=True)
        df['OnlineBackup'].replace(['No', 'Yes', 'No internet service'], [0, 1, 2], inplace=True)
        df['DeviceProtection'].replace(['No', 'Yes', 'No internet service'], [0, 1, 2], inplace=True)
        df['TechSupport'].replace(['No', 'Yes', 'No internet service'], [0, 1, 2], inplace=True)
        df['StreamingTV'].replace(['No', 'Yes', 'No internet service'], [0, 1, 2], inplace=True)
        df['StreamingMovies'].replace(['No', 'Yes', 'No internet service'], [0, 1, 2], inplace=True)
        df['Contract'].replace(['Month-to-month', 'One year', 'Two year'], [0, 1, 2], inplace=True)
        df['PaperlessBilling'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], [0, 1, 2, 3], inplace=True)
        df['Churn'].replace(['No', 'Yes'], [0, 1], inplace=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], downcast="float")
        df.sample(frac=1)
        data = df.to_numpy()
        data_x = (data[:, 0:-1] / np.amax(np.abs(data[:, 0:-1]), axis=0))
        data_y = data[:, -1]
        data_y = np.where(data_y == 0, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "philippine":
        df = pd.read_csv("datasets/philippine_train.data", header=None)
        data = df.to_numpy()
        data_x = (data / np.amax(np.abs(data), axis=0))
        df = pd.read_csv("datasets/philippine_train.solution", header=None)
        data_y = df.to_numpy().astype(int).reshape(-1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

#data_x, data_y, data_x_test, data_y_test = get_dataset("credit_card")
