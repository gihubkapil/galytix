import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.models import KeyedVectors


class PhraseSimilarity:
    def __init__(self, model_path):
        # Load the Word2Vec model
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


    def get_phrase_vectors(self, phrases):
        """
        Accepts a list of phrases and returns the corresponding list of phrase vectors.
        """
        phrase_vectors = []
        for phrase in phrases:
            # Split the phrase into words
            words = phrase.split()

            # Sum up the word vectors for the words that exist in the Word2Vec model
            word_vectors = []
            for word in words:
                if word in self.model:
                    word_vectors.append(self.model[word])

            if not word_vectors:  # If none of the words are in the model, return a zero vector
                phrase_vectors.append(np.zeros(self.model.vector_size))
                continue

            # Average the word vectors
            phrase_vector = np.mean(word_vectors, axis=0)

            # Normalize the vector (L2 normalization)
            norm = np.linalg.norm(phrase_vector)
            if norm != 0:
                phrase_vector /= norm

            phrase_vectors.append(phrase_vector)

        return phrase_vectors

    def batch_similarity(self, phrases):
        """
        Calculate pairwise similarity for all phrases in the list.
        """
        phrase_vectors = self.get_phrase_vectors(phrases)
        similarity_matrix = cosine_similarity(phrase_vectors)
        return similarity_matrix

    def closest_match(self, input_phrase, phrases):
        """
        Find the closest match to the input_phrase in the list of phrases using cosine similarity.
        """
        input_vector = self.get_phrase_vectors([input_phrase])[0]
        phrase_vectors = self.get_phrase_vectors(phrases)

        similarities = cosine_similarity([input_vector], phrase_vectors)
        most_similar_idx = np.argmax(similarities)
        closest_phrase = phrases[most_similar_idx]
        similarity_score = similarities[0][most_similar_idx]

        return closest_phrase, similarity_score