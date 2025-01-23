import streamlit as st

st.title('🎬Box Office Revenue Predictor')

st.info('This app can predict a box office revenue by using the genre of the movie and other related features')
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

group_labels = ['Domestic Revenue']
array = df['domestic_revenue'].to_numpy()
ar = [array]
fig = ff.create_distplot(ar , group_labels ,bin_size=[.1, .25, .5])

with st.expander('Data Visualization'):
  st.plotly_chart(fig , use_container_width=True)

for col in ['genres']:
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])

for col in ['distributor', 'MPAA']:
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])

ab = ['title','domestic_revenue' ]
features = df.drop(ab, axis=1)
target = df.domestic_revenue.values

X_train, X_val, Y_train, Y_val = train_test_split(features,target,test_size=0.2,random_state=22)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = XGBRegressor()
model.fit(X_train, Y_train)

train_preds = model.predict(X_train)
print('Training Error : ', mae(Y_train, train_preds))

val_preds = model.predict(X_val)
print('Validation Error : ', mae(Y_val, val_preds))

with st.sidebar:
	
	st.header('Input Features')
	title = st.text_input("Movie title", "Enter Movie Name")
	pro = st.radio(
	    "Choose the Production Company :",
	    ['Warner Bros.', 'Disney', 'Sony', 'Universal', 'Paramount'],
	)
	
	if pro == "Warner Bros.":
	    dis = 4
	elif pro == "Disney" :
	    dis = 0
	elif pro == "Sony" :
	    dis = 2
	elif pro == "Universal":
	    dis = 3	
	elif pro == "Paramount" :
	    dis = 1

	mpaa = st.radio(
	    "Chose the Production Company :",
	    ['R', 'G', 'NC', 'PG-13', 'PG'],
	)
	
	if mpaa == 'R':
	    mp = 4
	elif mpaa == 'G' :
	    mp = 0
	elif mpaa == 'PG' :
	    mp = 2
	elif mpaa == 'PG-13':
	    mp = 3	
	elif mpaa == 'NC' :
	    mp = 1


	genre = st.radio(
	    "Choose the Production Company :",
	    ['Animation', 'Action', 'Horror', 'Comedy', 'Drama', 'Thriller'],
	)
	
	if genre == "Animation":
	    gen = 1
	elif genre == "Action" :
	    gen = 0
	elif genre == "Horror" :
	    gen = 4
	elif genre == "Comedy":
	    gen = 2	
	elif genre == "Drama" :
	    gen = 3
	elif genre == "Thriller" :
	    gen = 5

	open_T = st.slider('Opening Theatre' , 10 ,4500, 2263)  
	release_D = st.slider('Release Date' , 1, 180 ,90)

data = {
	'title':title,
	'distributor':pro,
	'opening_theaters':open_T,
	'MPAA' : mpaa, 
	'genres' : genre,
       'release_days': release_D
	}
input_df = pd.DataFrame(data,index[0])
input_df
