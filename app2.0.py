import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
st.set_page_config(page_title='Titanic Survival Prediction', page_icon=":ship:", layout='wide')
# Load the dataset
df = pd.read_csv('combined_data.csv')

# Data Cleaning - Removing Null Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Splitting the dataset into the Training set and Test set
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Encoding Categorical Data
cols_to_encode = ['Sex', 'Embarked']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cols_to_encode)], remainder='passthrough')
X_train_enc = pd.DataFrame(ct.fit_transform(X_train))
X_test_enc = pd.DataFrame(ct.transform(X_test))

st.title('Titanic Survival Prediction :ship:')
st.markdown("""
Predict whether a passenger would have survived the Titanic disaster based on various factors.
""")

# Sidebar - Model Selection
st.sidebar.header('Select Model')
model = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes", "Decision Tree", "Random Forest"])

if model == "Logistic Regression":
    clf = LogisticRegression()
elif model == "K-Nearest Neighbors":
    clf = KNeighborsClassifier()
elif model == "Support Vector Machine":
    clf = SVC(kernel='linear')
elif model == "Naive Bayes":
    clf = GaussianNB()
elif model == "Decision Tree":
    clf = DecisionTreeClassifier()
elif model == "Random Forest":
    clf = RandomForestClassifier(n_estimators=10)
else:
    st.write("Select a model")

if st.sidebar.button("Train Model"):
    with st.spinner('Training Model...'):
        clf.fit(X_train_enc, y_train)
    st.success('Model Trained Successfully!')


if model != "":
    clf.fit(X_train_enc, y_train)
    y_pred = clf.predict(X_test_enc)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.write("Accuracy: ", accuracy)
    st.write("Precision: ", precision)
    st.write("Recall: ", recall)
    st.write("F1 Score: ", f1)
st.title('Enter the details of the passenger to predict the survival')
# User Input
sex = st.radio("Select Gender", ("male", "female"))
age = st.slider("Select Age", 0, 100, 30)
fare = st.number_input("Enter Fare", value=50)
embarked = st.selectbox("Select Embarked", ('C', 'Q', 'S'))
class_ = st.radio("Select Class", (1, 2, 3))
sibsp = st.slider("Select Number of Siblings/Spouses", 0, 8, 0)
parch = st.slider("Select Number of Parents/Children", 0, 6, 0)

# Mapping user input to DataFrame format
user_input = pd.DataFrame({'Sex': [sex], 'Age': [age], 'Fare': [fare], 'Embarked': [embarked], 'Pclass': [class_], 'SibSp': [sibsp], 'Parch': [parch]})
user_input_enc = pd.DataFrame(ct.transform(user_input))

if st.button("Predict"):
  prediction = clf.predict(user_input_enc)
  with st.spinner("Predicting..."):
      time.sleep(2)
  if prediction[0] == 1:
      st.success("Congratulations! You would have survived the Titanic disaster.")
      st.snow()
      st.balloons()
  else:
      st.error("Unfortunately, you would not have survived the Titanic disaster.")