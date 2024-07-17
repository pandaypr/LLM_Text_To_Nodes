from flask import Flask, request, render_template, redirect, url_for, flash
from models import load_model, classifier_zero, filter_known_nodes

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load models
classifier = load_model()

# Known nodes list
KNOWN_NODES = [
    "OnClick", "Delay", "Navigate", "FetchData", "DisplayModal", "Reduce", "Log", 
    "CacheData", "Show", "OnKeyPress", "Highlight", "OnMouseEnter", "OnMouseLeave", 
    "Filter", "Sort", "Update", "Console", "Alert", "Assign", "SendRequest", 
    "Save", "Delete", "PlaySound", "PauseSound", "StopSound", "Branch", "Map", 
    "GroupBy", "Merge", "Split", "Hide", "CloseModal", "Tooltip", "RenderChart", 
    "StoreData", "UpdateData", "DeleteData", "CacheData"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process form data
        text_input = request.form['text_input']

        if not text_input:
            flash('Please provide some input text.')
            return redirect(url_for('index'))

        # Classify text
        threshold = 0.80
        labels, scores = classifier_zero(classifier, text_input, KNOWN_NODES, multi_class=True)
        detected_labels = [label for label, score in zip(labels, scores) if score >= threshold]
        filtered_labels = filter_known_nodes([(label, 1.0) for label in detected_labels], KNOWN_NODES, threshold=threshold)
        results = [('Submitted Text', text_input, [label for label, score in filtered_labels])]
        return render_template('result.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
