import streamlit as st

st.title(' 🤖 Machine Learning App')

st.info('This app builds a Machine Learning Model')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('boxoffice.csv')
  df
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)
df.drop('budget', axis=1, inplace=True)
df.dropna(inplace=True)

