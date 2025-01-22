import streamlit as st

st.title(' 🤖 Machine Learning App')

st.info('This app builds a Machine Learning Model')
import pandas as pd
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)
df.drop('budget', axis=1, inplace=True)
df.dropna(inplace=True)
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('boxoffice.csv')
  df
  st.write('**X**')
  x = df.drop(['title','domestic_revenue'],axis=1 , inplace=True)
  x
  st.write('**Y**')
  y = df['domestic_revenue']
  y

  
  

