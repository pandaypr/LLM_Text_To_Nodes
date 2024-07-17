import json
import sys
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Add the parent directory of the test directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import load_model, classifier_zero, filter_known_nodes 

# List of known nodes for classification
KNOWN_NODES = [
    "OnClick", "Delay", "Navigate", "FetchData", "DisplayModal", "Reduce", "Log", 
    "CacheData", "Show", "OnKeyPress", "Highlight", "OnMouseEnter", "OnMouseLeave", 
    "Filter", "Sort", "Update", "Console", "Alert", "Assign", "SendRequest", 
    "Save", "Delete", "PlaySound", "PauseSound", "StopSound", "Branch", "Map", 
    "GroupBy", "Merge", "Split", "Hide", "CloseModal", "Tooltip", "RenderChart", 
    "StoreData", "UpdateData", "DeleteData", "CacheData"
]

def calculate_metrics(detected_sequence, expected_sequence):
    """
    Calculate precision, recall, and F1 score based on the detected and expected sequences.

    Args:
    detected_sequence (list of str): The sequence of nodes detected by the model.
    expected_sequence (list of str): The correct sequence of nodes from the test case.

    Returns:
    dict: A dictionary containing the precision, recall, and F1 score.
    """
    detected_set = set(detected_sequence)
    expected_set = set(expected_sequence)

    precision = len(detected_set & expected_set) / len(detected_set) if detected_set else 0 
    recall = len(detected_set & expected_set) / len(expected_set) if expected_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_test_cases(test_cases, classifier, known_nodes=KNOWN_NODES):
    """
    Evaluate the model's performance on a set of test cases.

    Args:
    test_cases (list of dict): A list of test cases where each test case is a dictionary containing
                               a 'prompt' and 'sequence_of_nodes'.
    classifier (pipeline): The classifier model to use for detecting nodes.
    known_nodes (list): A list of known nodes to filter detected nodes against.

    Returns:
    dict: A dictionary containing the average precision, recall, and F1 score across all test cases.
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    for test_case in test_cases:
        prompt = test_case['prompt']
        expected_sequence = test_case['sequence_of_nodes']
        
        labels, scores = classifier_zero(classifier, prompt, known_nodes, multi_class=True)
        detected_labels = [label for label, score in zip(labels, scores) if score >= 0.7]
        filtered_labels = filter_known_nodes([(label, 1.0) for label in detected_labels], known_nodes, threshold=0.7)
        detected_sequence = [label for label, score in filtered_labels]
        
        metrics = calculate_metrics(detected_sequence, expected_sequence)
        
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
    
    average_metrics = {
        'average_precision': sum(precisions) / len(precisions) if precisions else 0,
        'average_recall': sum(recalls) / len(recalls) if recalls else 0,
        'average_f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0
    }
    
    return average_metrics

if __name__ == '__main__':
    # Load test cases from the JSON file
    testcases_path = os.path.join(os.path.dirname(__file__), 'testcases.json')
    with open(testcases_path, 'r') as file:
        test_cases = json.load(file)

    # Load the classifier model
    classifier = load_model()

    # Evaluate the model on the test cases
    metrics = evaluate_test_cases(test_cases, classifier) 
    print(metrics)
