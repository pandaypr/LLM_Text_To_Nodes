import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from sentence_transformers import SentenceTransformer, util

def load_model():
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(task='zero-shot-classification', model=model, tokenizer=tokenizer, device=device)
    return classifier

def classifier_zero(classifier, sequence: str, labels: list, multi_class: bool):
    outputs = classifier(sequence, labels, multi_label=multi_class)
    return outputs['labels'], outputs['scores']

def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def filter_known_nodes(detected_labels, known_nodes, threshold=0.8):
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
