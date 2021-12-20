# import libraries required
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

st.title('Spam message dectection')
message = st.text_input('Enter your message here')
# importing dataset from github
df = pd.read_table('https://raw.githubusercontent.com/diazonic/Machine-Learning-using-sklearn/master/Datasets/spam.tsv')

# Splitting the data into x & y
x = df['message'].values
y = df['label'].values

# Splitting the dataset in ratio of 75:25
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

#performing normalization
vect  = CountVectorizer(stop_words='english')
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

#Using naive bayes algo 
model = MultinomialNB()
model.fit(x_train_vect,y_train)
message1 = vect.transform([message])
y_pred = model.predict(message1)
if st.button('PREDICT'):
    st.title(y_pred)
