import unittest
import json
import sys
import os

# Add the parent directory of the test directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        """
        Set up the test client for the Flask application.
        """
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        """
        Test the index page to ensure it loads correctly.
        """
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
        self.assertIn(b'LLM based Text to Node Application', result.data)

    def test_prompt_processing(self):
        """
        Test the processing of input prompts and verify the expected output nodes are present in the response.
        """
        testcases_path = os.path.join(os.path.dirname(__file__), 'testcases.json')
        with open(testcases_path, 'r') as f:
            test_cases = json.load(f)

        for case in test_cases:
            prompt = case['prompt']
            expected_nodes = case['sequence_of_nodes']

            result = self.app.post('/', data={'text_input': prompt})
            self.assertEqual(result.status_code, 200)

            response_data = result.data.decode('utf-8')
            missing_nodes = []
            for index, node in enumerate(expected_nodes, start=1):
                formatted_node = f"{index}. [{node}]"
                if formatted_node not in response_data:
                    missing_nodes.append(formatted_node)

            if missing_nodes:
                missing_nodes_str = ", ".join(missing_nodes)
                self.fail(f"Missing nodes for prompt '{prompt}': {missing_nodes_str}")

    def test_no_text_input(self):
        """
        Test the scenario where no text input is provided, expecting a redirect.
        """
        result = self.app.post('/', data={'text_input': ''})
        self.assertEqual(result.status_code, 302)

    def test_invalid_text_input(self):
        """
        Test the application with invalid input, ensuring it still processes the request correctly.
        """
        prompt = '!!!@@@###'
        result = self.app.post('/', data={'text_input': prompt})
        self.assertEqual(result.status_code, 200)
        self.assertIn('Submitted Text', result.data.decode('utf-8'))

if __name__ == '__main__':
    unittest.main()
