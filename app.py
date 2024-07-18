from flask import Flask, request, render_template, redirect, url_for, flash
from models import load_model, classifier_zero, filter_known_nodes

app = Flask(__name__)

# Load models
classifier = load_model()

# List of known nodes for classification from the given pdf file
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
    """
    Handle the main route of the application.
    If the request method is POST, process the form data and classify the input text.
    If the request method is GET, render the index page.
    """
    if request.method == 'POST':
        # Process form data
        text_input = request.form['text_input']

        if not text_input:
            # Flash a message and redirect if no input is provided
            flash('Please provide some input text.')
            return redirect(url_for('index'))

        # Classify text
        threshold = 0.80  # Set the threshold for classification scores
        labels, scores = classifier_zero(classifier, text_input, KNOWN_NODES, multi_class=True)
        detected_labels = [label for label, score in zip(labels, scores) if score >= threshold]
        # Filter known nodes based on the threshold
        filtered_labels = filter_known_nodes([(label, 1.0) for label in detected_labels], KNOWN_NODES, threshold=threshold)
        # Prepare the results for rendering
        results = [('Submitted Text', text_input, [label for label, score in filtered_labels])]
        return render_template('result.html', results=results)
    
    # Render the index page for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
