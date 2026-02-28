import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ======================
# Load and clean dataset
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    # Parse JSON-like columns (genres + keywords)
    def extract_names(x):
        try:
            items = ast.literal_eval(x)
            return " ".join([d["name"] for d in items])
        except:
            return ""

    df["genres"] = df["genres"].apply(extract_names)
    df["keywords"] = df["keywords"].apply(extract_names)

    # Fill missing overviews/taglines
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")

    # Create a "soup" = overview + genres + keywords + tagline
    df["soup"] = (
        df["overview"] + " " + df["genres"] + " " + df["keywords"] + " " + df["tagline"]
    )

    return df

df = load_data()

# ======================
# Build TF-IDF similarity
# ======================
@st.cache_resource
def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_similarity(df)

# ======================
# Recommendation function
# ======================
def recommend(title, top_n=5):
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]  # exclude itself
    movie_indices = [i[0] for i in sim_scores]
    return df[["title", "genres", "vote_average"]].iloc[movie_indices]

# ======================
# Streamlit UI
# ======================
st.title("🎬 Movie Recommendation System (Content-Based)")
st.write("Search for a movie and get similar recommendations!")

# Dropdown to select movie
movie_list = df["title"].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, top_n=5)
    if recommendations is None:
        st.error("❌ Movie not found in dataset.")
    else:
        st.subheader(f"Because you liked **{selected_movie}**, you may also like:")
        for idx, row in recommendations.iterrows():
            st.write(f"**{row['title']}**  | 🎭 Genres: {row['genres']} | ⭐ Rating: {row['vote_average']}")
