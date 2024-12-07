
import argparse
from src.phrase_similarity import PhraseSimilarity


class PhraseSimilarityCLI:
    def __init__(self):
        # Set up the CLI interface
        self.parser = argparse.ArgumentParser(description="Phrase Similarity CLI")
        self.parser.add_argument('-m', '--model', required=True, help="Path to the Word2Vec model file (binary)")
        self.parser.add_argument('-f', '--file', help="Path to a CSV or text file containing phrases")
        self.parser.add_argument('-p', '--phrase', help="Input phrase for closest match query")
        self.parser.add_argument('-b', '--batch', action='store_true',
                                 help="Calculate and display similarity matrix for all phrases")

    def load_phrases_from_file(self, file_path):
        # Assuming phrases are in a text file (one phrase per line)
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            phrases = [line.strip() for line in file.readlines()]
        return phrases

    def run(self):
        # Parse arguments
        args = self.parser.parse_args()

        # Create PhraseSimilarity object
        phrase_similarity = PhraseSimilarity(args.model)

        if args.file:
            # Load phrases from file
            phrases = self.load_phrases_from_file(args.file)

            if args.batch:
                # If batch mode is enabled, calculate pairwise similarities
                similarity_matrix = phrase_similarity.batch_similarity(phrases)
                print("Cosine Similarity Matrix:")
                print(similarity_matrix)
            elif args.phrase:
                # If a phrase is provided, find the closest match
                closest_phrase, similarity_score = phrase_similarity.closest_match(args.phrase, phrases)
                print(f"Closest match to '{args.phrase}':")
                print(f"Phrase: {closest_phrase} with similarity score: {similarity_score}")
        else:
            print("Please provide a file with phrases using --file option.")


if __name__ == '__main__':
    cli = PhraseSimilarityCLI()
    cli.run()
