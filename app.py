#!/usr/bin/env python
# coding: utf-8

# # OPIM-607-201 Final Project
# ## Michael Kong
# ### December 2, 2022

# In[1]:


#pip install altair vega_datasets
#pip install streamlit
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st


# #### Q1: Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[2]:

s = st.file_uploader("social_media_usage", type=["csv", "json"], accept_multiple_files=False)


# #### Q2: Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[4]:

# Define clean_sm function
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)


# #### Q3: Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[7]:


#Create new dataframe with target and features
ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] >9,np.nan,s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan,s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 2, 1, 0),
    "age":np.where(s["age"] > 98, np.nan,s["age"])})

#ss.head()


# In[8]:


# Drop missing data
ss = ss.dropna()


# #### Q4: Create a target vector (y) and feature set (X)

# In[12]:


#Target (y) sm_li, (x) features of income, education, parent, married, female, and age
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female","age"]]


# ***

# #### Q5: Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[13]:


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=123)


# The 'X_train' object contains 80% of the data and contains the features (provided in question 3) used to predict the target (sm_li) when training my model. 
# 
# The 'X_test' object contains 20% of the data and contains the features used to test my model on unseen ss data to evaluate model performance. 
# 
# The 'y_train' object contains 80% of the the data and contains the target that will be predicted using the features when training my model. 
# 
# The 'y_test' contains 20% of the data and contains the target that will be predicted when testing my model on unseen data to evaluate performance.

# ***

# #### Q6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[14]:


#Instantiate a logistic regression model and set class_weight to balanced. 
lr = LogisticRegression(class_weight='balanced', random_state=0).fit(X, y)


# In[15]:


#Fit the model with the training data.
lr.fit(X_train, y_train)


# In[16]:


# Evaluate the model using the testing data. Use the model to make predictions.
y_pred = lr.predict(X_test)


# ***
import streamlit as st

st.title("Linkedin User Prediction App")
st.caption("The purpose of this app is to predict whether a person is a Linkedin user based on certain variables.")
st.caption("OPIM-607-201 Final - Michael Kong")


income = st.selectbox("What is your current income level ($)?", 
              options = ["---",
                         "Less than 10,000",
                         "10,000 - 19,999",
                         "20,000 - 29,999",
                         "30,000 - 39,999",
                         "40,000 - 49,999",
                         "50,000 - 74,999",
                         "75,000 - 99,999",
                         "100,000 - 149,999",
                         "150,000 or more"])
if income == "Less than 10,000":
     income = 1
elif income == "10,000 - 19,999":
     income = 2
elif income == "20,000 - 29,999":
     income = 3
elif income == "30,000 - 39,999":
     income = 4
elif income == "40,000 - 49,999":
     income = 5
elif income == "50,000 - 74,999":
     income = 6
elif income == "75,000 - 99,999":
     income = 7
elif income == "100,000 - 149,999":
     income = 8
elif income == "150,000 or more":
     income = 9
else:
     income = 0


education = st.selectbox("What is your highest education level achieved?", 
              options = ["---",
                         "Less than High School",
                         "High School Incomplete",
                         "High School Graduate",
                         "Some College, No Degree",
                         "Two-Year Associate Degree from College/University",
                         "Four-Year College or University Bachelor's Degree (e.g., BS, BA, AB)",
                         "Some Postgraduate or Professional Schooling, No Postgraduate Degree",
                         "Postgraduate or Profressional Degree (e.g., MA, MS, PhD, MD, JD)"])
if education == "Less than High School":
     education = 1
elif education == "High School Incomplete":
     education = 2
elif education == "High School Graduate":
     education = 3
elif education == "Some College, No Degree":
     education = 4
elif education == "Two-Year Associate Degree from College/University":
     education = 5
elif education == "Four-Year College or University Bachelor's Degree (e.g., BS, BA, AB)":
     education = 6
elif education == "Some Postgraduate or Professional Schooling, No Postgraduate Degree":
     education = 7
elif education == "Postgraduate or Profressional Degree (e.g., MA, MS, PhD, MD, JD)":
     education = 8
else:
     education = 0


parent = st.selectbox("Are you a parent of a child under 18 living in your home?", 
              options = ["---",
                         "Yes",
                         "No"])
if parent == "Yes":
     parent = 1
else:
     parent = 0


married = st.selectbox("What is your current marrital status?", 
              options = ["---",
                         "Married",
                         "Living with a partner",
                         "Divorced",
                         "Separated",
                         "Widowed",
                         "Never been married"])
if married == "Married":
     married = 1
else:
     married = 0


gender = st.selectbox("What is your gender?", 
              options = ["---",
                         "Male",
                         "Female",
                         "Other",
                         "Don't know",
                         "Refused"])
if gender == "Female":
     gender = 1
else:
     gender = 0

age = st.slider(label="What is your current age?", 
           min_value=1,
           max_value=98,
           value=98)



# New data for features (person 1): "income", "education", "parent", "married", "female","age"
person = [income,education,parent,married,gender,age]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class
probs = lr.predict_proba([person])


# Print predicted class and probability
st.write(f"Linkedin User Prediction (Yes=1; No=0): {predicted_class[0]}") # 0=non-user, 1=user
st.write(f"Probability that this person is a Linkedin User: {probs[0][1]}")
