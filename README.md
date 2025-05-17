import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained models and data
@st.cache(allow_output_mutation=True)
def load_data():
    # Example: load preprocessed DataFrame
    df = pd.read_csv('preprocessed_movies.csv')
    # Load user_features and kmeans model
    with open('user_features.pkl', 'rb') as f:
        user_features = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    return df, user_features, kmeans

df, user_features, kmeans = load_data()

st.title("AI-Driven Movie Recommendation & Matchmaking System")

# User Input
user_id = st.number_input("Enter your User ID:", min_value=int(df['userId'].min()),
                          max_value=int(df['userId'].max()), value=int(df['userId'].min()))

if st.button("Get Recommendations"):
    # 1. Predict cluster and find similar users
    cluster_label = kmeans.predict(user_features.loc[[user_id]])[0]
    peers = user_features[user_features['cluster'] == cluster_label].index.tolist()
    peers.remove(user_id)

    # 2. Compute cosine similarity
    sims = cosine_similarity(user_features.loc[[user_id]].drop('cluster', axis=1),
                             user_features.drop('cluster', axis=1))[0]
    sim_users = [uid for uid, score in zip(user_features.index, sims) if score > 0.8 and uid != user_id]

    # 3. Recommend top movies based on peers' ratings
    peer_ratings = df[df['userId'].isin(sim_users)]
    top_movies = peer_ratings.groupby('movieId')['scaled_rating']\
                       .mean().sort_values(ascending=False).head(5).index.tolist()
    movie_titles = df[df['movieId'].isin(top_movies)]['title'].unique().tolist()

    # Display
    st.subheader("Recommended Movies:")
    for title in movie_titles:
        st.write(f"- {title}")

    st.subheader("Users with Similar Taste:")
    for uid in sim_users[:5]:
        st.write(f"- User {uid}")

# Footer
st.markdown("---")
st.write("Developed by Kailashini M | Tagore Engineering College")

