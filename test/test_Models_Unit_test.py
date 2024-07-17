import unittest
import sys
import os
import torch
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer


# Add the parent directory of the test directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import (
    load_model, classifier_zero, load_similarity_model, filter_known_nodes
)

class TestModels(unittest.TestCase):

    def test_load_model(self):
        """
        Test the loading of the classification model.
        """
        classifier = load_model()
        self.assertIsNotNone(classifier)

    def test_classifier_zero(self):
        """
        Test the zero-shot classification function on a given test case from the assignment.
        """
        classifier = load_model()
        sequence = "Navigate to a new page after a delay of 3 seconds when the user clicks a button"
        labels = ["OnClick", "Delay", "Navigate"]
        output_labels, scores = classifier_zero(classifier, sequence, labels, multi_class=True)
        self.assertEqual(len(output_labels), len(labels))
        self.assertEqual(len(scores), len(labels))

    def test_load_similarity_model(self):
        """
        Test the loading of the similarity model.
        """
        similarity_model = load_similarity_model()
        self.assertIsNotNone(similarity_model)

    def test_filter_known_nodes(self):
        """
        Test the filtering of known nodes based on similarity scores.
        """
        detected_labels = [("TestLabel", 0.95), ("Label", 0.85)]
        known_nodes = ["TestLabel", "KnownLabel"]
        filtered_labels = filter_known_nodes(detected_labels, known_nodes, threshold=0.9)
        expected_output = [("TestLabel", 0.95)]
        self.assertEqual(filtered_labels, expected_output)

if __name__ == '__main__':
    unittest.main()
