import streamlit as st

st.title(' 🤖 Machine Learning App')

st.info('This app builds a Machine Learning Model')
import pandas as pd
with st.expander('Data'):
  df = pd.read_csv('boxoffice.csv')
  df
  to_remove = ['world_revenue', 'opening_revenue']
  df.drop(to_remove, axis=1, inplace=True)
  df.drop('budget', axis=1, inplace=True)
  df.dropna(inplace=True)

