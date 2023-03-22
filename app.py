# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:02:53 2022

@author: Hp
"""

import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("Survival_Prediction.sav","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('train.csv')
#X = dataset.iloc[:, [2, 4, 5, 6, 7, 9]].values
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)
def predict_note_authentication(Pclass, Sex, Age, SibSp, Parch, Fare):
  output= model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare]])
  print("Survived", output)
  if output==[1]:
    prediction="Survived"
  else:
    prediction="not Survived"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Internship Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Survived or Not")
    Pclass = st.number_input("Pclass",1,3)
    Age = st.number_input("Insert Age",10,100)
    Sex = st.selectbox('Sex',('Male', 'Female'))
    if Sex == 'Male':
        Sex = 1
    else:
        Sex = 0
    SibSp = st.number_input("SibSp",0,1)
    Parch = st.number_input("Parch",0,2)
    Fare = st.number_input("Fare", 1,100)
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Pclass, Sex, Age, SibSp, Parch, Fare)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Nitish Nama")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()