import streamlit as st

st.title(' 🤖 Machine Learning App')

st.info('This app builds a Machine Learning Model')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('boxoffice.csv')
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)
df.drop('budget', axis=1, inplace=True)
df.dropna(inplace=True)
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('boxoffice.csv')
  df
  st.write('**X**')
  ab = ['title','domestic_revenue']
  x = df.drop(ab,axis=1)
  x
  st.write('**Y**')
  y = df.domestic_revenue
  y
for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].astype(str).str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')
features = ['domestic_revenue', 'opening_theaters', 'release_days']
for col in features:
  df[col] = df[col].apply(lambda x: np.log10(x))

with st.expander('Data Visualization'):
  for i, col in enumerate(features):
     plt.subplot(1, 3, i+1)
     sb.distplot(df[col])
  plt.tight_layout()
  plt.show()


  

