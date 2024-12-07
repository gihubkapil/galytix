import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine, euclidean

app = Flask(__name__)

# Global variable to store the pre-calculated phrase vectors and similarity matrix
phrase_vectors = []
similarity_matrix = []
phrases_df = None
model = None


# Load the Word2Vec model from file
def load_model():
    global model
    model = KeyedVectors.load_word2vec_format('/Users/kapil/Downloads/vectors.csv', binary=False)


# Function to calculate phrase vector by averaging word vectors
def get_phrase_vector(phrase):
    words = phrase.split()
    word_vectors = [model[word] for word in words if word in model]

    if not word_vectors:
        return np.zeros(model.vector_size)

    phrase_vector = np.mean(word_vectors, axis=0)
    norm = np.linalg.norm(phrase_vector)
    if norm != 0:
        phrase_vector /= norm

    return phrase_vector


# Function to calculate cosine similarity between two vectors
def calculate_similarity(phrase_vector_1, phrase_vector_2, metric='cosine'):
    if metric == 'cosine':
        return cosine(phrase_vector_1, phrase_vector_2)
    elif metric == 'euclidean':
        return euclidean(phrase_vector_1, phrase_vector_2)
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")


# Pre-calculate phrase vectors and similarity matrix on app startup
def pre_calculate_similarity_matrix():
    global phrase_vectors, similarity_matrix, phrases_df
    phrases_df = pd.read_csv('/Users/kapil/Downloads/phrases.csv')

    # Calculate phrase vectors for all phrases
    phrase_vectors = [get_phrase_vector(phrase) for phrase in phrases_df['Phrases']]

    # Calculate similarity matrix (cosine similarity between all pairs of phrases)
    num_phrases = len(phrase_vectors)
    similarity_matrix = np.zeros((num_phrases, num_phrases))

    for i in range(num_phrases):
        for j in range(i, num_phrases):
            similarity_score = calculate_similarity(phrase_vectors[i], phrase_vectors[j], 'cosine')
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score  # Symmetric matrix


# API endpoint to get the closest phrase match
@app.route('/closest-match', methods=['POST'])
def closest_match():
    input_phrase = request.json.get('phrase')

    if not input_phrase:
        return jsonify({'error': 'No phrase provided'}), 400

    # Get the input phrase vector
    input_vector = get_phrase_vector(input_phrase)

    closest_phrase = None
    min_similarity = float('inf')

    # Find the closest match using the pre-calculated similarity matrix
    for idx, phrase in enumerate(phrases_df['phrase']):
        similarity = calculate_similarity(input_vector, phrase_vectors[idx], 'cosine')

        if similarity < min_similarity:
            min_similarity = similarity
            closest_phrase = phrase

    return jsonify({'closest_phrase': closest_phrase, 'similarity': min_similarity})


# Initialize model and pre-calculate similarity matrix
if __name__ == '__main__':
    load_model()  # Load the Word2Vec model
    pre_calculate_similarity_matrix()  # Pre-calculate phrase similarity matrix
    app.run(debug=True)
