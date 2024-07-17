import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from sentence_transformers import SentenceTransformer, util

def load_model():
    """
    Load the zero-shot classification model from Hugging Face's Transformers library.
    
    Returns:
        pipeline: A zero-shot classification pipeline.
    """
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer, device=device)
    return classifier

def classifier_zero(classifier, sequence: str, labels: list, multi_class: bool):
    """
    Perform zero-shot classification on the given sequence.

    Args:
        classifier (pipeline): The zero-shot classification pipeline.
        sequence (str): The input text to classify.
        labels (list): A list of candidate labels.
        multi_class (bool): Whether to allow multiple labels per sequence.

    Returns:
        tuple: A tuple containing the labels and their corresponding scores.
    """
    outputs = classifier(sequence, labels, multi_label=multi_class)
    return outputs['labels'], outputs['scores']

def load_similarity_model():
    """
    Load the SentenceTransformer model for semantic similarity tasks.
    
    Returns:
        SentenceTransformer: A SentenceTransformer model.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def filter_known_nodes(detected_labels, known_nodes, threshold=0.8):
    """
    Filter detected labels against known nodes based on similarity scores.

    Args:
        detected_labels (list): A list of tuples containing detected labels and their scores.
        known_nodes (list): A list of known nodes to filter against.
        threshold (float): The similarity threshold for filtering.

    Returns:
        list: A list of filtered labels that match the known nodes.
    """
    model = load_similarity_model()
    known_node_embeddings = model.encode(known_nodes, convert_to_tensor=True)

    filtered_labels = []
    for label, score in detected_labels:
        if score < threshold:
            continue
        label_embedding = model.encode(label, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(label_embedding, known_node_embeddings)
        max_score, max_index = torch.max(cosine_scores, dim=1)
        if max_score.item() >= threshold:
            filtered_labels.append((known_nodes[max_index.item()], score))

    return filtered_labels
