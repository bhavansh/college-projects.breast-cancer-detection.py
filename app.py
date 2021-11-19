import streamlit as st

import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score

import statistics

st.title('Breast Cancer Dataset Analysis')
st.sidebar.title('Breast Cancer Dataset Analysis')

st.markdown(
    'This application is used to learn about use of various classifiers on Breast Cancer dataset')
st.sidebar.markdown(
    'This application is used to learn about use of various classifiers on Breast Cancer dataset')

DATA_URL = ("breast_cancer.csv")


@st.cache(persist=True)
def load():
    data = pd.read_csv(DATA_URL)
    df = pd.read_csv("breast_cancer.csv")
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    # Impute the only categorical column
    df["diagnosis"].replace(to_replace=dict(M=1, B=0), inplace=True)
    df.dropna()
    return df


def show_evaluation_metrics(confusion_metrics):
    tp = confusion_metrics[1, 1]
    fn = confusion_metrics[1, 0]
    fp = confusion_metrics[0, 1]
    tn = confusion_metrics[0, 0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn))) /
                                             ((tp/(tp+fp))+(tp/(tp+fn))))))


data = load()

# select = st.sidebar.selectbox(
#     'Visualization Type', ['Histogram', 'Pie chart', 'PCA'], key='1')

classifier_name = st.sidebar.selectbox(
    "Select the classifier", ("KNN", "SVM", "Random Forest", "Logistic Regression"))


def get_dataset(data):
    X = data[["radius_mean", "texture_mean", "smoothness_mean",
              "compactness_mean", "concavity_mean"]]
    y = data["diagnosis"]
    #y = y.reshape(len(y),1)
    return X, y


X, y = get_dataset(data)
st.write('Shape of Dataset', X.shape)
st.write('Number of classes', len(np.unique(y)))


def add_parameter_ui(class_name):
    params = dict()
    if class_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K

    elif class_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    elif class_name == 'Random Forest':
        max_depth = st.sidebar.slider("Max_Depth", 2, 100)
        n_estimators = st.sidebar.slider("N-Estimators", 1, 1000)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params


params = add_parameter_ui(classifier_name)


def get_classifier(class_name, params):
    if class_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=params["K"])

    elif class_name == "SVM":
        classifier = SVC(C=params["C"])
    elif class_name == "Logistic Regression":
        classifier = LogisticRegression()

    else:
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["max_depth"], random_state=1234)
    return classifier


classifier = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy}")
cm_XG = confusion_matrix(y_test, y_pred)
st.write('Confusion matrix: ', cm_XG)
