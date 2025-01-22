import streamlit as st

st.title(' 🤖 Machine Learning App')

st.info('This app builds a Machine Learning Model')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
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
  ab = ['title','domestic_revenue' ,'world_revenue', 'opening_revenue' ]
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

group_labels = ['Domestic Revenue']
array = df['domestic_revenue'].to_numpy()
ar = [array]
fig = ff.create_distplot(ar , group_labels ,bin_size=[.1, .25, .5])

with st.expander('Data Visualization'):
  st.plotly_chart(fig , use_container_width=True)


vectorizer = CountVectorizer()
vectorizer.fit(df['genres'])
features = vectorizer.transform(df['genres']).toarray()

genres = vectorizer.get_feature_names_out()
for i, name in enumerate(genres):
	df[name] = features[:, i]

df.drop('genres', axis=1, inplace=True) 
features = df.drop(['title', 'domestic_revenue'], axis=1)
target = df['domestic_revenue'].values

for col in ['distributor', 'MPAA']:
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])

X_train, X_val, Y_train, Y_val = train_test_split(x,y,test_size=0.2,random_state=22)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = XGBRegressor()
model.fit(X_train, Y_train)

train_preds = model.predict(X_train)
print('Training Error : ', mae(Y_train, train_preds))

val_preds = model.predict(X_val)
print('Validation Error : ', mae(Y_val, val_preds))

									

