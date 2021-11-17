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
    df=pd.read_csv("breast_cancer.csv")
    df = df.drop(['Unnamed: 32','id'],axis = 1)
    # Impute the only categorical column
    df["diagnosis"].replace(to_replace = dict(M = 1, B = 0), inplace = True)
    df.dropna()
    return df

def show_evaluation_metrics(confusion_metrics):
    tp = confusion_metrics[1,1]
    fn = confusion_metrics[1,0]
    fp = confusion_metrics[0,1]
    tn = confusion_metrics[0,0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))



data = load()

# select = st.sidebar.selectbox(
#     'Visualization Type', ['Histogram', 'Pie chart', 'PCA'], key='1')

classifier_name = st.sidebar.selectbox(
    "Select the classifier", ("KNN", "SVM", "Random Forest", "Logistic Regression"))


def get_dataset(data):
    X=data[["radius_mean","texture_mean","smoothness_mean","compactness_mean","concavity_mean"]]
    y=data["diagnosis"]
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

#Visualize
# st.markdown("Number of cover of each type")
# if select == "Histogram":
#     fig = px.bar(cover_count, x='Cover Type',
#                  y='Values', color='Values', height=500)
#     st.plotly_chart(fig)
# elif select == "Pie chart":
#     fig = px.pie(cover_count, names='Cover Type', values='Values')
#     st.plotly_chart(fig)
# else:
#     pca = PCA(2)
#     X_projected = pca.fit_transform(X)
#     x1 = X_projected[:, 0]
#     x2 = X_projected[:, 1]
#     plt.figure()
#     plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.colorbar()
#     st.pyplot()

# if st.checkbox("ANALYZE DATASET"):
#     if st.sidebar.checkbox("Preview Dataset"):
#         if st.sidebar.button("Head"):
#             st.write(data.head())
#         elif st.sidebar.button("Tail"):
#             st.write(data.tail())
#         else:
#             number = st.sidebar.slider("Select No of Rows", 1, data.shape[0])
#             st.write(data.head(number))

#     # show column names
#     if st.checkbox("Show Column Names"):
#         st.write(data.columns)

#     # show dimensions
#     if st.checkbox("Show Dimensions"):
#         st.write(data.shape)

#     # show summary
#     if st.checkbox("Show Summary"):
#         st.write(data.describe())

#     # show missing values
#     if st.checkbox("Show Missing Values"):
#         st.write(data.isna().sum())

#     # Select a column to treat missing values
#     col_option = st.selectbox(
#         "Select Column to treat missing values", data.columns)

#     # Specify options to treat missing values
#     missing_values_clear = st.selectbox("Select Missing values treatment method", (
#         "Replace with Mean", "Replace with Median", "Replace with Mode"))

#     if missing_values_clear == "Replace with Mean":
#         replaced_value = data[col_option].mean()
#         st.write("Mean value of column is :", replaced_value)
#     elif missing_values_clear == "Replace with Median":
#         replaced_value = data[col_option].median()
#         st.write("Median value of column is :", replaced_value)
#     elif missing_values_clear == "Replace with Mode":
#         replaced_value = data[col_option].mode()
#         st.write("Mode value of column is :", replaced_value)

#     Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
#     if Replace == "Yes":
#         data[col_option] = data[col_option].fillna(replaced_value)
#         st.write("Null values replaced")
#     elif Replace == "No":
#         st.write("No changes made")

#     # To change datatype of a column in a dataframe
#     # display datatypes of all columns
#     if st.checkbox("Show datatypes of the columns"):
#         st.write(data.dtypes)

#     # visualization
#     # scatter plot
#     col1 = st.selectbox('Which feature on x?', data.columns)
#     col2 = st.selectbox('Which feature on y?', data.columns)
#     fig = px.scatter(data, x=col1, y=col2)
#     st.plotly_chart(fig)

#     # correlartion plots
#     if st.checkbox("Show Correlation plots with Seaborn"):
#         st.set_option('deprecation.showPyplotGlobalUse', False)
#         st.write(sns.heatmap(data.corr()))
#         st.pyplot()

# if st.sidebar.checkbox('Show Raw Data', False):
#     st.write(data)


# output
# if st.checkbox('OUTPUT ANALYSIS', False):
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy}")
cm_XG = confusion_matrix(y_test, y_pred)
st.write('Confusion matrix: ', cm_XG)