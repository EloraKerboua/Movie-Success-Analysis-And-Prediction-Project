import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
# Ensure you have the correct path to your data
final_cleaned_data = pd.read_csv('final_cleaned_data.csv')

# Data processing
final_cleaned_data['Budget'] = final_cleaned_data['Budget'].replace('N/A', 0).astype(float)
final_cleaned_data['imdbRating'] = final_cleaned_data['imdbRating'].replace('N/A', '0').str.replace(',', '.').astype(float)
final_cleaned_data['BoxOffice'] = final_cleaned_data['BoxOffice'].replace('N/A', 0).astype(float)

# Feature and target variables
features = final_cleaned_data[['Budget', 'Director', 'Actors', 'Genre']]
target = final_cleaned_data[['imdbRating', 'BoxOffice']]

# Column transformer
column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Director', 'Actors', 'Genre']),
    ('num', StandardScaler(), ['Budget'])
])

# IMDb Rating model
imdb_pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('model', LinearRegression())
])
imdb_pipeline.fit(features, target['imdbRating'])

# BoxOffice model
box_office_pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('model', LinearRegression())
])
box_office_pipeline.fit(features, target['BoxOffice'])

# Streamlit app
st.title('Movie Rating and Box Office Predictor')

# Input widgets
budget = st.number_input('Budget ($)', min_value=100000, max_value=1000000000, step=50000)
director = st.text_input('Director')
actors = st.text_input('Actors')
genre = st.text_input('Genre')

if st.button('Predict'):
    input_data = pd.DataFrame({'Budget': [budget], 'Director': [director], 'Actors': [actors], 'Genre': [genre]})
    
    # Predict IMDb Rating
    imdb_pred = imdb_pipeline.predict(input_data)[0]
    # Predict Box Office
    boxoffice_pred = box_office_pipeline.predict(input_data)[0]
    
    # Display predictions
    st.write(f"Predicted IMDb Rating: {imdb_pred:.2f}")
    st.write(f"Predicted Box Office Revenue: ${boxoffice_pred:.2f}")
    
    # Plotting the predictions
    plt.figure(figsize=(14, 6))
    
    # Actual IMDb ratings vs Budget
    plt.subplot(1, 2, 1)
    plt.scatter(final_cleaned_data['Budget'], final_cleaned_data['imdbRating'], alpha=0.5, label='Actual Data')
    plt.scatter(budget, imdb_pred, color='red', label='Prediction', s=100)
    plt.title('IMDb Rating vs Budget')
    plt.xlabel('Budget ($)')
    plt.ylabel('IMDb Rating')
    plt.ylim(0, 10)  # Ensure the y-axis is limited to 0-10 for IMDb ratings
    plt.legend()
    
    # Actual Box Office vs Budget
    plt.subplot(1, 2, 2)
    plt.scatter(final_cleaned_data['Budget'], final_cleaned_data['BoxOffice'], alpha=0.5, label='Actual Data')
    plt.scatter(budget, boxoffice_pred, color='red', label='Prediction', s=100)
    plt.title('Box Office vs Budget')
    plt.xlabel('Budget ($)')
    plt.ylabel('Box Office Revenue ($)')
    plt.legend()
    
    st.pyplot(plt)
