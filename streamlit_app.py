import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('boxoffice.csv')
    df2 = df.copy()
    df.drop(['world_revenue', 'opening_revenue', 'budget'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Convert columns to numeric
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['genres', 'distributor', 'MPAA']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df, df2

# Train model
@st.cache_resource
def train_model(features, target):
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=22)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    return model, scaler, mse(Y_train, train_preds), r2(Y_train, train_preds)

# Load and prepare data
df, df2 = load_data()
x = df.drop(['title', 'domestic_revenue'], axis=1)
y = df.domestic_revenue
model, scaler, mse_error, r2_error = train_model(x, y)

# App UI
st.title('🎬 Box Office Revenue Predictor')
st.info('This app predicts domestic box office revenue using movie features.')

# Show data
with st.expander('Data'):
    st.write('*Raw Data*')
    st.write(df2.head(20))
    st.write('*X (Features)*')
    st.write(x.head(10))
    st.write('*Y (Target)*')
    st.write(y.head(10))

# Data visualization
group_labels = ['Domestic Revenue']
array = df['domestic_revenue'].to_numpy()
fig = ff.create_distplot([array], group_labels, bin_size=[10000000])

with st.expander('Data Visualization'):
    st.plotly_chart(fig, use_container_width=True)

# Model metrics
with st.expander('Model Metrics'):
    st.write('Mean Squared Error:', mse_error)
    st.write('R2 Score:', r2_error)

# Sidebar inputs
with st.sidebar:
    st.header('Input Features')
    title = st.text_input("Movie Title", "Example Movie")
    pro = st.radio("Production Company", ['Warner Bros.', 'Disney', 'Sony', 'Universal', 'Paramount'])
    mpaa = st.radio("MPAA Rating", ['R', 'G', 'NC', 'PG-13', 'PG'])
    genre = st.radio("Genre", ['Animation', 'Action', 'Horror', 'Comedy', 'Drama', 'Thriller'])
    open_T = st.slider('Opening Theatres', 10, 4500, 2263)
    release_D = st.slider('Release Days', 1, 180, 90)
    load = st.button('Predict Revenue')

# Make prediction
if load:
    pro_dict = {'Warner Bros.': 4, 'Disney': 0, 'Sony': 2, 'Universal': 3, 'Paramount': 1}
    mpaa_dict = {'R': 4, 'G': 0, 'NC': 1, 'PG': 2, 'PG-13': 3}
    genre_dict = {'Animation': 1, 'Action': 0, 'Horror': 4, 'Comedy': 2, 'Drama': 3, 'Thriller': 5}

    input_data = {
        'distributor': pro_dict[pro],
        'opening_theaters': open_T,
        'MPAA': mpaa_dict[mpaa],
        'genres': genre_dict[genre],
        'release_days': release_D
    }

    input_df = pd.DataFrame(input_data, index=[0])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader('Predicted Domestic Revenue')
    st.success(f'$ {prediction:,.0f}')

    with st.expander("Input Summary"):
        st.write('*Original Inputs:*')
        st.write(pd.DataFrame({
            'title': [title],
            'distributor': [pro],
            'MPAA': [mpaa],
            'genre': [genre],
            'opening_theaters': [open_T],
            'release_days': [release_D]
        }))
        st.write('*Encoded Inputs:*')
        st.write(input_df)
