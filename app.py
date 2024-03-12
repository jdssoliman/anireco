import streamlit as st
import joblib
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
# import seaborn as sns

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy
from surprise import SVDpp
from surprise.model_selection import cross_validate

import joblib
import random
import warnings
import time

# from IPython.display import display, Image

import requests
from bs4 import BeautifulSoup
import time


# Suppress specific RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


loaded_model = joblib.load('collaborative_filtering_model.pkl')
loaded_model_item = joblib.load('collaborative_filtering_model_item.pkl')
df_utility_loaded = pd.read_pickle("df_utility.pkl")
df_rand_user_loaded = pd.read_pickle("df_rand_user.pkl")
df_anime_loaded = pd.read_pickle("df_anime.pkl")
tfidf_df_loaded = pd.read_pickle("tfidf_df.pkl")

def get_user_ratings_streamlit(anime_dict_to_rate):
    ratings = {}
    for anime_id, names in anime_dict_to_rate.items():

        while True:  # This loop ensures the user provides a valid rating
            try:
                # link = f'https://myanimelist.net/anime/{anime_id}'
                # time.sleep(1)
                # response = requests.get(link)
                # soup = BeautifulSoup(response.content, 'html.parser')
                # img_link = soup.find('img', class_='ac')['data-src']
                # # URL of the image
                # image_url = img_link

                # # names, image_url = info[0], info[1]  # Assuming info contains names and pre-fetched image URL
        
                # # Display the anime image and names
                # st.image(image_url, caption=f"{names[0]} / {names[1]}", width=300)
                
                # Slider for rating
                rating = st.slider(f"Rate '{names[0]}' (0 to 10):", 1, 10, key=anime_id)
                
                # Check if the rating is within the valid range
                if 0 <= rating <= 10:
                    # Store the rating in the dictionary
                    ratings[anime_id] = rating
                    break  # Exit the loop once a valid rating is provided
                else:
                    print("Please enter a rating between 1 and 10.")
            except ValueError:  # Handle the case where the input is not an integer
                print("Invalid input. Please enter a number.")


def vectorize(ratings,df_utility_loaded):
    # Vectorize the user_ratings in df_utility
    df_user_rated = pd.DataFrame(columns=df_utility_loaded.columns)
    # Fill the DataFrame with NaNs initially
    df_user_rated.loc[0, :] = np.nan
    
    # Update the DataFrame with ratings from the dictionary
    for anime_id, rating in ratings.items():
        if anime_id in df_user_rated.columns:
            df_user_rated.loc[0, anime_id] = rating
            
    return df_user_rated

def custom_cosine_distance(x, y):
    # Ensure x and y are numpy arrays of type float to safely use np.isnan
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    non_null_indices = ~np.isnan(x) & ~np.isnan(y)

    if not np.any(non_null_indices):
        return np.nan

    x_matching = x[non_null_indices]
    y_matching = y[non_null_indices]

    # Check if x_matching and y_matching have enough data points
    if len(x_matching) < 2 or len(y_matching) < 2:
        return np.nan

    correlation, _ = pearsonr(x_matching, y_matching)
    distance = 1 - correlation

    if np.isnan(distance):
        return 0.0
    return distance

def find_closest_row(df, target_row):
    min_distance = np.inf
    closest_row_index = None

    # Ensure target_row is properly converted to numeric dtype numpy array
    target_row_array = np.asarray(target_row, dtype=float)

    # Iterate over all rows in the DataFrame
    for index, row in df.iterrows():
        # Convert row to numpy array with dtype float for compatibility
        row_array = np.asarray(row, dtype=float)
        
        distance = custom_cosine_distance(target_row_array, row_array)

        if distance < min_distance:
            min_distance = distance
            closest_row_index = index

    return closest_row_index

def get_reco_cf(df_top_user, user, algo, df_anime):
    # this is original anime reco, without filters, so that no bias or subjective reasoning integrated
    user_items = df_top_user[df_top_user['user_id'] == user]['anime_id'].tolist()
    all_items = df_top_user['anime_id'].unique()
    not_rated_items = [item for item in all_items if item not in user_items]
    recommendations = [algo.predict(user, item) for item in not_rated_items]
    recommendations.sort(key=lambda x: x.est, reverse=True)
    recommended_items = [pred.iid for pred in recommendations]
    df_anime_reco = pd.DataFrame(columns=df_anime.columns)
    
    for id in recommended_items:
        row_reco = df_anime[df_anime['anime_id']==id]
        df_anime_reco = pd.concat([df_anime_reco, row_reco], ignore_index=True)
        
    return df_anime_reco
    
def filter_recos(df, n=10):
    to_exclude_genres = [
        'Boys Love',
        'Ecchi',
        'Erotica',
        'Girls Love',
        'UNKNOWN'
    ]
    to_exclude_ratings = [
        # 'R - 17+ (violence & profanity)',
        'R+ - Mild Nudity',
        'Rx - Hentai',
        'UNKNOWN']

    to_exclude_type = [
        'Special',
        'Music',
        'UNKNOWN'
    ]
    def should_include(genres_list):
        return not any(genre in to_exclude_genres for genre in genres_list)

    to_exclude_rank = ['0.0', 'UNKNOWN']
    df['Genres'] = df['Genres'].str.split(', ')
    df = df[df['Genres'].apply(should_include)]
    df = df[~df['Rating'].isin(to_exclude_ratings)]
    df = df[~df['Type'].isin(to_exclude_type)]
    df = df[df["Synopsis"] != "No description available for this anime."]
    df = df[~df['Rank'].isin(to_exclude_rank)]
    df = df[df['Popularity']!=0]
    df['Rank'] =  pd.to_numeric(df['Rank'], downcast='integer')
    df = df[~df['Rank']<100]
    return df.head(n)

def recommend_user(df_rand_user, user, loaded_model, df_anime, rated_r=False):
    df_anime_reco_user = get_reco_cf(
        df_rand_user, user, loaded_model, df_anime)
    if rated_r:
        return df_anime_reco_user.iloc[:10][['Name', 'English name']]
    else:
        return filter_recos(df_anime_reco_user)[['Name', 'English name']]

def recommend_item(df_rand_user, user, loaded_model_item, df_anime, rated_r=False):
    df_anime_reco_item = get_reco_cf(
        df_rand_user, user, loaded_model_item, df_anime)
    if rated_r:
        return df_anime_reco_item.iloc[:10][['Name', 'English name']]
    else:
        return filter_recos(df_anime_reco_item)[['Name', 'English name']]

def recommend_content(df_utility, tfidf_df, user, df_anime, rated_r=False):
    user_profile_467 = compute_user_profile_agg_numeric(
        df_utility, tfidf_df, user)
    list_anime_467 = recommend_agg_numeric(
        df_utility, tfidf_df, user_profile_467, user)
    df_reco_anime_content = recommended_anime_content(list_anime_467, df_anime)
    if rated_r:
        return df_reco_anime_content.iloc[:10][['Name', 'English name']]
    else:
        return filter_recos(df_reco_anime_content)[['Name', 'English name']]

def recommend_agg_numeric(df_utility, df_item_profiles, user_profile, user):
    nan_idx = np.isnan(df_utility.loc[user])
    items = df_item_profiles.loc[nan_idx]
    ratings = sorted(
        [
            (i, cosine(item, user_profile))
            for i, item in items.iterrows()
            if cosine(item, user_profile) > 0
        ],
        key=lambda x: (x[1], x[0]),
    )
    return [i for i, _ in ratings]
from scipy.spatial.distance import cosine

def compute_user_profile_agg_numeric(df_utility, df_item_profiles, user):
    user_ratings = df_utility.loc[user]
    mean_rating = user_ratings.mean()
    centered_ratings = user_ratings - mean_rating
    weighted_profiles = df_item_profiles.mul(centered_ratings, axis=0)
    user_profile = weighted_profiles.mean()
    return user_profile

def recommended_anime_content(list_id, df_anime):
    df_anime_reco = pd.DataFrame(columns=df_anime.columns)
    
    for id in list_id:
        row_reco = df_anime[df_anime['anime_id']==id]
        df_anime_reco = pd.concat([df_anime_reco, row_reco], ignore_index=True)
        
    return df_anime_reco

def sumirolap(jed_rating, df_utility_loaded, df_rand_user_loaded, loaded_model, loaded_model_item, df_anime_loaded, tfidf_df_loaded):
    jed_vector = vectorize(jed_rating, df_utility_loaded)
    user = find_closest_row(df_utility_loaded, jed_vector)
    user_reco = recommend_user(df_rand_user_loaded, user, loaded_model, df_anime_loaded, rated_r=False)
    item_reco = recommend_item(df_rand_user_loaded, user, loaded_model_item, df_anime_loaded, rated_r=False)
    content_reco = recommend_content(df_utility_loaded, tfidf_df_loaded, user, df_anime_loaded, rated_r=False)
    return user_reco, item_reco, content_reco

def streamlit_app():
    st.title('Anime Recommendation System')

    # Placeholder for a dictionary of anime to rate
    anime_dict_to_rate = {1535: ('Death Note', 'Death Note'),
             20583: ('Haikyuu!!', 'Haikyu!!'),
             20: ('Naruto', 'Naruto'),
             530: ('Bishoujo Senshi Sailor Moon', 'Sailor Moon'),
             763: ('Zoids', 'Zoids')}
                
   
    # jed_rating = get_user_ratings_streamlit(anime_dict_to_rate)
    
    st.image("death_note.jpeg")
    death_note_rating = st.number_input('Death Note', step=1, format="%d", value=1)
    st.image("haikyu.jpeg")
    haikyuu_rating = st.number_input('Haikyu!!', step=1, format="%d", value=1)
    st.image("naruto.jpeg")
    naruto_rating = st.number_input('Naruto', step=1, format="%d", value=1)
    st.image("sailor_moon.jpeg")
    sailor_moon_rating = st.number_input('Sailor Moon', step=1, format="%d", value=1)
    st.image("zoids.jpeg")
    zoids_rating = st.number_input('Zoids', step=1, format="%d", value=1)



    if st.button('Submit Ratings and Recommend!'):
        jed_rating = {
    1535: death_note_rating,
    20583: haikyuu_rating,
    20: naruto_rating,
    530: sailor_moon_rating,
    763: zoids_rating
    }

        user_reco, item_reco, content_reco = sumirolap(jed_rating, df_utility_loaded, df_rand_user_loaded, loaded_model, loaded_model_item, df_anime_loaded, tfidf_df_loaded)
        
        # Display Recommendations
        st.subheader('User-based Recommendations')
        st.write(user_reco)
        
        st.subheader('Item-based Recommendations')
        st.write(item_reco)
    
        st.subheader('Content-based Recommendations')
        st.write(content_reco)
        
    # if st.button('Show recommendations'):
        

if __name__ == '__main__':
    streamlit_app()