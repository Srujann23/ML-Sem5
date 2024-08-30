import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#Loading the dataset

# Diabetes

diabetes_dataset = pd.read_csv('diabetes.csv')

X=diabetes_dataset.drop(columns= 'Outcome',axis=1)
Y=diabetes_dataset['Outcome']

scaler=StandardScaler()
std_data=scaler.fit_transform(X)

X=std_data
Y=diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)

# from sklearn.model_selection import GridSearchCV

# para = {'C': [0.1, 1, 10, 100],
#  'gamma': [1, 0.1, 0.01, 0.001],
#  'kernel': ['linear','rbf','poly','sigmoid']}
# clf=GridSearchCV(svm.SVC(),para,cv=5,refit=True,verbose=3)
clf=svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)

X_train_prediction= clf.predict(X_train)
trian_data_accuracy=accuracy_score(X_train_prediction,Y_train)

X_test_prediction= clf.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

# loading data

# diabetes_model = pickle.load(open('D:/Sem-5/ML/Innovative_project/diabetes_model.sav','rb'))
# heart_model = pickle.load(open('D:/Sem-5/ML/Innovative_project/heart_model.sav','rb'))

#Heart Model

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')
heart_data.head()

heart_data['target'].value_counts()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model =DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# sidebar 

with st.sidebar:

    selected=option_menu('Multiple Disease Prediction System',

                          ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Skin cancer detection'], 

                            icons=['activity','heart-fill','person-fill-slash','file-earmark-medical-fill'],

                            default_index=0)

if(selected=='Diabetes Prediction'):
  st.title('Diabetes Prediction.')

  col1, col2, col3 = st.columns(3)

  with col1:
      Pregnancies = st.text_input('Number of Pregnancies')
        
  with col2:
      Glucose = st.text_input('Glucose Level')
    
  with col3:
      BloodPressure = st.text_input('Blood Pressure value')
    
  with col1:
      SkinThickness = st.text_input('Skin Thickness value')
    
  with col2:
      Insulin = st.text_input('Insulin Level')
    
  with col3:
      BMI = st.text_input('BMI value')
    
  with col1:
      DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
  with col2:
      Age = st.text_input('Age of the Person')



#prediction

diab_diagnosis = ''
# diab_prediction=0  
    # creating a button for Prediction
    
if st.button('Diabetes Test Result'):
        # diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        input_data=(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)

        data_array=np.asarray(input_data)

        data_reshaped=data_array.reshape(1,-1)

        std_reshaped_data=scaler.transform(data_reshaped)

        # print(std_reshaped_data)

        diab_prediction = clf.predict(std_reshaped_data) 
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
          st.success(diab_diagnosis)
        else:
          diab_diagnosis = 'The person is not diabetic'
          st.success(diab_diagnosis)

if(selected=='Heart Disease Prediction'):
  st.title('Heart Disease Prediction.')
    
  col1, col2, col3 = st.columns(3)
    
  with col1:
        age = st.text_input('Age')
        
  with col2:
        sex = st.text_input('Sex')
        
  with col3:
        cp = st.text_input('Chest Pain types')
        
  with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
  with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
  with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
  with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
  with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
  with col3:
        exang = st.text_input('Exercise Induced Angina')
        
  with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
  with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
  with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
  with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    # code for Prediction
heart_diagnosis = ''
# heart_prediction=0
    # creating a button for Prediction
if st.button('Heart Disease Test Result'):
      input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)

      # change the input data to a numpy array
      input_data_as_numpy_array= np.asarray(input_data,dtype=np.float64)

      # reshape the numpy array as we are predicting for only on instance
      input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

      heart_prediction = model.predict(input_data_reshaped)
      # print(prediction)                        
         
      if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
          st.success(heart_diagnosis)

      else:
          heart_diagnosis = 'The person does not have any heart disease'
          st.success(heart_diagnosis)

#skin cancer

if(selected=='Skin cancer detection'):
  st.title('Skin cancer detection')
