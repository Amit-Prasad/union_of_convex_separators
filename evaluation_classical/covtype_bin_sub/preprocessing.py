import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset(name):
    if name == "covtype_bin_sub":
        class_0 = 1
        class_1 = 2
        data = pd.read_csv('../../datasets/covtype.data', sep=",").sample(frac=1).values
        n = data.shape[0]
        x_train, x_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.30, random_state= 1)
        x_train = x_train / np.amax(abs(x_train), axis=0)

        class_indices = np.where((y_train == class_0) | (y_train == class_1))
        data_x = x_train[class_indices[0], :]
        data_y = y_train[class_indices]
        data_y = np.where(data_y == class_0, 0, 1)
        class_indices = np.where((y_test == class_0) | (y_test == class_1))
        data_x_test = x_test[class_indices[0], :]
        data_y_test = y_test[class_indices]
        data_y_test = np.where(data_y_test == class_0, 0, 1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.97, random_state=1)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "churn":
        df = pd.read_csv("../../datasets/Churn_Modelling.xls")
        df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
        df['Geography'].replace(['France', 'Germany', 'Spain'], [0, 1, 2], inplace=True)
        df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
        df = df[(df != '?').all(axis=1)]
        df.sample(frac=1)
        data = df.to_numpy()
        data_x = (data[:, 0:-1] / np.amax(np.abs(data[:, 0:-1]), axis=0))
        data_y = data[:, -1]
        data_y = np.where(data_y == 0, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "telco_churn":
        df = pd.read_csv("../../datasets/TelcoCustomerChurn.xls")
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

    if name == "santander_sub":
        df = pd.read_csv("../../datasets/santander_train.csv")
        data_y = df['target'].values
        df.drop(['ID_code', 'target'], axis=1, inplace=True)
        df.sample(frac=1)
        data_x = df.to_numpy()
        data_x = (data_x / np.amax(np.abs(data_x), axis=0))
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.95, random_state=1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "spambase":
        df = pd.read_csv("../../datasets/spambase.data", header=None)
        df = df[(df != '?').all(axis=1)]
        cols = df.columns
        for col in cols:
            df[col] = df[col].astype(float)
        df.sample(frac=1)
        data = df.to_numpy()
        data_x = (data[:, 0:-1] / np.amax(np.abs(data[:, 0:-1]), axis=0))
        data_y = data[:, -1].astype(int)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "shoppers":
        df = pd.read_csv("../../datasets/online_shoppers_intention.csv")
        df['Month'].replace(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
        df['VisitorType'].replace(['New_Visitor', 'Other', 'Returning_Visitor'], [0, 1, 2], inplace=True)
        df['Weekend'].replace([False, True], [0, 1], inplace=True)
        df['Revenue'].replace([False, True], [0, 1], inplace=True)
        df.sample(frac=1)
        df.columns = df.loc[0]
        df = df.drop(0)
        data = df.to_numpy()
        data_x = (data[:, 0:-1] / np.amax(np.abs(data[:, 0:-1]), axis=0))
        data_y = data[:, -1].astype(int)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "philippine":
        df = pd.read_csv("../../datasets/philippine_train.data", header=None)
        data = df.to_numpy()
        data_x = (data / np.amax(np.abs(data), axis=0))
        df = pd.read_csv("../../evaluation/data/philippine_train.solution", header=None)
        data_y = df.to_numpy().astype(int).reshape(-1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "diabetes":
        data = pd.read_csv("../../datasets/diabetes.csv", skiprows=1, header = None).values
        data_x = data[:, 0:-1]/np.amax(np.abs(data[:, 0:-1]), axis=0)
        data_y = data[:, -1].astype(int)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "ionosphere":
        df = pd.read_csv('../../datasets/ionosphere.data', sep=",").sample(frac=1)
        df.replace(['b', 'g'], [0, 1], inplace=True)
        data = df.values
        data = np.delete(data, 1, axis=1)
        data_x = data[:, 0:-1]/np.amax(np.abs(data[:, 0:-1]), axis=0)
        data_y = data[:, -1].astype(int)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == "breast_cancer":
        df = pd.read_csv('../../datasets/breast-cancer.data', header = None).sample(frac=1)
        df = df[(df != '?').all(axis=1)]
        df.replace(["no-recurrence-events", "recurrence-events"], [0, 1], inplace=True)
        df.replace(["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"], [0, 1, 2, 3, 4, 5], inplace=True)
        df.replace(["ge40", "lt40", "premeno"], [0, 1, 2], inplace=True)
        df.replace(["left", "right"], [0, 1], inplace=True)
        df.replace(["no", "yes"], [0, 1], inplace=True)
        df.replace(["central", "left_low", "left_up", "right_up", "right_low"], [0, 1, 2, 3, 4], inplace=True)
        df[3].replace(["0-4", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "5-9", "50-54"], [0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10], inplace = True)
        df[4].replace(["0-2", "12-14", "15-17", "24-26", "25-29", "3-5", "6-8", "9-11"], [0, 1, 2, 3, 4, 5 , 6, 7], inplace=True)
        cols = df.columns
        for col in cols:
            df[col] = df[col].astype(float)
        data = df.values
        data_y = data[:, 0].astype(int)
        data_x = data[:, 1:]
        data_x = data_x / np.amax(np.abs(data_x), axis=0)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

