import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the processed TV data
tv_data_processed = pd.read_csv('tv_data_processed.csv')

# Recommendation function
def get_recommendations(title, cosine_sim):
    escaped_title = re.escape(title)  # Escape special characters for regex matching
    matches = tv_data_processed[tv_data_processed['name'].str.contains(escaped_title, case=False)]
    
    if matches.empty:
        st.write(f"No match found for the title: {title}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    
    id = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations excluding itself
    
    similar_titles = tv_data_processed.iloc[[i[0] for i in sim_scores]]  # Get TV show data
    return similar_titles  # Return a DataFrame

# Compute cosine similarity
count_vect = CountVectorizer()
count_matrix = count_vect.fit_transform(tv_data_processed['aggregated_text'])
cosine_sim_c = cosine_similarity(count_matrix, count_matrix)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📺 ViewWise, A TV Show Recommendation System")

st.subheader("Like a TV Show? Select it for Recommendations:")

# Dropdown list for TV show selection
selected_show = st.selectbox("Choose a TV show:", tv_data_processed['name'].unique())

# Show recommendations on button click
if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_show, cosine_sim_c)
    
    if not recommendations.empty:  # Check if the recommendations DataFrame is not empty
        st.write("Recommended TV Shows:")

        # Display recommendations in a grid of 5 columns and 2 rows
        num_columns = 5
        num_rows = 2
        total_recommendations = min(len(recommendations), num_columns * num_rows)
        
        for row_idx in range(num_rows):
            columns = st.columns(num_columns)
            for col_idx in range(num_columns):
                index = row_idx * num_columns + col_idx
                if index < total_recommendations:
                    # Fetch the row of the current recommendation
                    row = recommendations.iloc[index]

                    # Display the TV show image from the 'image' column
                    columns[col_idx].image(row['image'], use_column_width=True)

                    # Display the name, IMDb link, rating value, and rating count
                    columns[col_idx].markdown(
                        f"<p style='text-align:center; margin-top:-10px;'><strong>{row['name']}</strong></p>",
                        unsafe_allow_html=True)
                    columns[col_idx].markdown(
                        f"<p style='text-align:center;'>"
                        f"<a href='{row['url']}' target='_blank'>IMDb Link</a></p>",
                        unsafe_allow_html=True)
                    columns[col_idx].markdown(
                        f"<p style='text-align:center;'>Rating: {row['rating_value']} ⭐ "
                        f"({row['rating_count']} ratings)</p>",
                        unsafe_allow_html=True)

st.text("Last Updated: 5/9/2024 (Updates Yearly)")
st.text("(Include TV Shows IMDB Rating > 7.5, No. Ratings > 50,000)")
st.text("By: Hong Kai")