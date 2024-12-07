import pytest
import numpy as np
from src.phrase_similarity import PhraseSimilarity


# Test loading word vectors from CSV
@pytest.fixture
def mock_model():
    """Fixture to provide a mock model dictionary."""
    return {
        'deep': np.array([0.12, 0.34, 0.56]),
        'learning': np.array([0.11, 0.35, 0.57]),
        'machine': np.array([0.13, 0.33, 0.58]),
    }


# Test phrase vector calculation
def test_get_phrase_vectors(mock_model):
    phrase_similarity = PhraseSimilarity(csv_path=None)
    phrase_similarity.model = mock_model  # Mock model

    # Test for a single phrase
    phrases = ["deep learning"]
    vectors = phrase_similarity.get_phrase_vectors(phrases)

    # Expected vector is the mean of the "deep" and "learning" word vectors
    expected_vector = np.mean([mock_model['deep'], mock_model['learning']], axis=0)
    np.testing.assert_almost_equal(vectors[0], expected_vector)

    # Test for a non-existent word in the phrase (should return zero vector)
    phrases = ["nonexistent"]
    vectors = phrase_similarity.get_phrase_vectors(phrases)
    np.testing.assert_almost_equal(vectors[0], np.zeros_like(expected_vector))


# Test batch similarity calculation
def test_batch_similarity(mock_model):
    phrase_similarity = PhraseSimilarity(csv_path=None)
    phrase_similarity.model = mock_model  # Mock model

    phrases = ["deep learning", "machine learning", "artificial intelligence"]
    similarity_matrix = phrase_similarity.batch_similarity(phrases)

    # The similarity matrix should be of size (3, 3)
    assert similarity_matrix.shape == (3, 3)

    # Verify diagonal values (self-similarity should be 1)
    for i in range(3):
        assert np.isclose(similarity_matrix[i, i], 1.0)


# Test closest match finding
def test_closest_match(mock_model):
    phrase_similarity = PhraseSimilarity(csv_path=None)
    phrase_similarity.model = mock_model  # Mock model

    input_phrase = "deep learning"
    phrases = ["machine learning", "deep learning", "artificial intelligence"]

    closest_phrase, similarity_score = phrase_similarity.closest_match(input_phrase, phrases)

    # The closest phrase should be "deep learning" with a high similarity score
    assert closest_phrase == "deep learning"
    assert similarity_score > 0.9
