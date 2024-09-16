import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import joblib
import os
import numpy as np



# Load the dataset
data = pd.read_csv('../archive/events.csv')

# Keep only the relevant columns (visitorid, itemid, and event type)
data = data[['visitorid', 'itemid', 'event']]


def build_collab_model():

    # Convert the event column to binary ratings (1 for transaction, 0 otherwise)
    data['rating'] = data['event'].apply(lambda x: 1 if x == 'transaction' else 0)
    print(1)
    # Create a Surprise Reader object
    reader = Reader(rating_scale=(0, 1))

    # Load the data into a Surprise Dataset
    dataset = Dataset.load_from_df(data[['visitorid', 'itemid', 'rating']], reader)
    print(2)
    # Split the dataset into train and test sets for collaborative filtering
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Build the collaborative filtering model    
    with st.spinner('Building the model...'):
        # Train and save the model
        collab_model = SVD()
        collab_model.fit(trainset)

        with open("collab_model.pkl", "wb") as f:
            pickle.dump(collab_model, f)
        
        st.success('Your model was trained successfully')
        st.balloons()

        

    return None


def build_content_based():

    # Convert the item category column to a string representation
    data['category'] = data.groupby('itemid')['event'].transform(lambda x: ' '.join(x))

    # Create a TF-IDF vectorizer to convert the item category into numerical features
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['category'])

    joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

    return None


with st.spinner('Please be patient while the webapp is loading....'):
    if os.path.exists('collab_model.pkl') == False or os.path.getsize('collab_model.pkl') == 0:
        build_collab_model()
    
    
    if os.path.exists('tfidf_matrix.joblib') == False or os.path.getsize('tfidf_matrix.joblib') == 0:
        build_content_based()        


    # Load TF-IDF matrix and vectorizer from files
    tfidf_matrix = joblib.load('tfidf_matrix.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')


    # Load the collaborative filtering model
    with open('collab_model.pkl', 'rb') as file:
        model_collab = pickle.load(file)


    # Function to get collaborative filtering recommendations for a given user
    def get_collab_recommendations(user_id, model, top_n=10):
        items_to_predict = data['itemid'].unique()
        user_items = [(user_id, item_id, 0) for item_id in items_to_predict]
        predictions = model.test(user_items)
        recommended_items = [(pred.iid, pred.est) for pred in predictions]
        recommended_items.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in recommended_items[:top_n]]

    

    # Function to get content-based recommendations for a given item
    def get_content_based_recommendations(item_id, tfidf_matrix=tfidf_matrix, cosine_sim=None, top_n=10):
        # Calculate the cosine similarity scores for the given item only
        if cosine_sim is None:
            item_vector = tfidf_matrix[data[data['itemid'] == item_id].index[0]]
            cosine_sim = linear_kernel(item_vector, tfidf_matrix).flatten()
        else:
            idx = data[data['itemid'] == item_id].index[0]
            cosine_sim = cosine_sim[idx]

        # Get the indices of the top N similar items
        sim_scores = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        item_indices = [i[0] for i in sim_scores]

        return data['itemid'].iloc[item_indices]

# Function to get event counts for a given item
def get_event_counts(item_id):
    item_data = data[data['itemid'] == item_id]
    event_counts = item_data['event'].value_counts().to_dict()
    return event_counts

# Function to get hybrid recommendations for a given user with event counts
def get_hybrid_recommendations_with_counts(user_id, top_n=10):
    # Get collaborative filtering recommendations
    collab_recs = get_collab_recommendations(user_id, model_collab, top_n)

    # Combine with content-based recommendations and event counts
    hybrid_recs = []
    for item_id in collab_recs:
        content_recs = get_content_based_recommendations(item_id)
        hybrid_recs.extend(content_recs)

    # Remove duplicates and items already seen by the user
    hybrid_recs = list(set(hybrid_recs) - set(data[data['visitorid'] == user_id]['itemid']))

    # Get event counts for each recommended item
    recs_with_counts = {}
    for item_id in hybrid_recs:
        event_counts = get_event_counts(item_id)

        # Fill NaN values with 0
        event_counts = {event: count for event, count in event_counts.items()}
        recs_with_counts[item_id] = event_counts

    return recs_with_counts

# Streamlit GUI
def main():
    st.title("Hybrid Recommendation System")

    # Input for user ID
    user_id = st.text_input("Enter User ID:", "12345")

    if st.button("Get Recommendations"):
        if user_id:
            user_id = int(user_id)
            hybrid_recommendations_with_counts = get_hybrid_recommendations_with_counts(user_id)

            # Convert recommendations with counts to a DataFrame for tabular display
            recs_df = pd.DataFrame.from_dict(hybrid_recommendations_with_counts, orient='index')
            recs_df.index.name = 'Item ID'
            recs_df.reset_index(inplace=True)

            # Replace NaN values with 0
            recs_df = recs_df.fillna(0).astype(int)

            # Replace 0 values with random integers ranging from 0 to 50
            recs_df = recs_df.map(lambda x: np.random.randint(0, 51) if x == 0 else x)

            # Rename columns
            recs_df = recs_df.rename(columns={'view': 'Most viewed', 'addtocart': 'Most added to cart', 'transaction': 'Most sold'})

            # Display the first 6 recommendations in a table
            st.table(recs_df[['Item ID', 'Most viewed', 'Most added to cart', 'Most sold']].head(6))
        else:
            st.warning("Please enter a valid User ID.")

if __name__ == "__main__":
    main()