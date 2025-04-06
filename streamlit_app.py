import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

@st.cache_data
def load_data():
    df = pd.read_csv('boxoffice.csv')
    df2 = df.copy()
    df2 = df2[df2['domestic_revenue'] >= 100000000]
    df = df[df['domestic_revenue'] >= 100000000]
    df.drop(['world_revenue', 'opening_revenue', 'budget','title'], axis=1, inplace=True)
    df.dropna(inplace=True)

    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].apply(lambda x: np.log10(x))
    for col in ['genres', 'distributor', 'MPAA']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df , df2

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


df , df2 = load_data()

x = df.drop(['title', 'domestic_revenue'], axis=1)
y = df.domestic_revenue

model, scaler, mse_error, r2_error = train_model(x, y)

st.title('🎬Box Office Revenue Predictor')
st.info('This app can predict domestic box office revenue by using the genre of the movie and other related features')

with st.expander('Data'):
    st.write('*Raw Data*')
    st.write(df2.head(20))
    st.write('*X*')
    st.write(x.head(10))
    st.write('*Y*')
    st.write(y.head(10))

group_labels = ['Domestic Revenue']
array = df['domestic_revenue'].to_numpy()
fig = ff.create_distplot([array], group_labels, bin_size=[.1, .25, .5])

with st.expander('Data Visualization'):
    st.plotly_chart(fig, use_container_width=True)

with st.expander('Metrics'):
    st.write('Mean Squared Error :' ,mse_error )
    st.write('R2 Error :' ,r2_error )

with st.sidebar:
    
        st.header('Input Features')
        title = st.text_input("Movie title", "Enter Movie Name")
        pro = st.radio("Choose the Production Company:", ['Warner Bros.', 'Disney', 'Sony', 'Universal', 'Paramount'])
        mpaa = st.radio("Choose MPAA:", ['R', 'G', 'NC', 'PG-13', 'PG'])
        genre = st.radio("Choose Genre:", ['Animation', 'Action', 'Horror', 'Comedy', 'Drama', 'Thriller'])
        open_T = st.slider('No. Of Theatres', 0.0, 10.0, 1.0)
        release_D = st.slider('Released Days', 0.0, 10.0, 1.0)
        load = st.button('Load Button')
        
if load :
    pro_dict = {'Warner Bros.': 4, 'Disney': 0, 'Sony': 2, 'Universal': 3, 'Paramount': 1}
    mpaa_dict = {'R': 4, 'G': 0, 'NC': 1, 'PG': 2, 'PG-13': 3}
    genre_dict = {'Animation': 1, 'Action': 0, 'Horror': 4, 'Comedy': 2, 'Drama': 3, 'Thriller': 5}
    
    data = {'title': title, 'distributor': pro, 'opening_theaters': open_T, 'MPAA': mpaa, 'genres': genre, 'release_days': release_D}
    input_df = pd.DataFrame(data, index=[0])
    data_en = {'distributor': pro_dict[pro], 'opening_theaters': open_T, 'MPAA': mpaa_dict[mpaa], 'genres': genre_dict[genre], 'release_days': release_D}
    input_en = pd.DataFrame(data_en, index=[0])

    with st.expander("Input Data"):
                st.write('*Data*')
                st.write(input_df)
                st.write('*Encoded Data*')
                st.write(input_en)
        
    input_scaled = scaler.transform(input_en)
    answer = model.predict(input_scaled)
    ans = 10 ** answer[0]
    st.subheader('Predicted Revenue')
    st.success('$ ' + "{:,}".format(ans))
