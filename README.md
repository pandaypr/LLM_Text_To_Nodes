# LLM Based Text to Node Application

This project is a Flask web application that uses a pre-trained language model to classify text inputs into known nodes. The application also includes unit tests to ensure the functionality of the models and the application itself.
The initial requirement was to use the google/codegemma-7b model or another model with approximately 7 billion parameters from Hugging Face. However, due to hardware limitations, this wasn't feasible.

## Model Selection and Challenges
The primary reason for choosing the facebook/bart-large-mnli model, which has approximately 406 million parameters, was the lack of access to a GPU capable of handling models larger models. Running inference with a 7 billion parameter model requires substantial computational resources, including a large GPU, which was not available for this project.

Attempts were made to use cloud-based services, such as Hugging Face's inference API, to handle the larger model. Unfortunately, these efforts were unsuccessful due to various limitations and constraints in the cloud service.

Additionally, efforts were made to fine-tune smaller models such as BART, T5, and GPT-2, which are around 1.5 billion parameters. However, their performance in text generation tasks was below that of the facebook/bart-large-mnli model used in this project. As a result, the facebook/bart-large-mnli model was selected as the most suitable alternative, providing a balance between performance and computational feasibility. This model enables efficient text classification while remaining within the capabilities of the available hardware.

## Similarity Model and Adjustable Threshold

In addition to the classification model, the application uses a similarity model based on 'all-MiniLM-L6-v2' and cosine similarity to compare the output with the expected nodes. This allows for a more flexible and accurate classification process. The threshold value for similarity can be adjusted in the app.py file, allowing users to change the amount of nodes generated based on their specific requirements.

## Project Structure
<pre><code>
project_root/
│
├── app.py                 # Main Flask application
├── models.py              # Model-related functions and loading mechanisms
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── README.md              # Project documentation
├── templates/             # HTML templates
│   ├── index.html
│   ├── layout.html
│   └── result.html
├── static/                # Static files (CSS, JS)
│   └── styles.css
└── test/                  # Unit tests
    ├── App_Unit_Test.py
    ├── metrics.py
    ├── Models_Unit_test.py
    └── testcases.json     # Test cases for unit tests
</code></pre>

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized setup)

### Installing Dependencies

1. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment:
    ```bash
    # On Windows
    .venv\Scripts\activate

    # On macOS/Linux
    source .venv/bin/activate
    ```

3. Install the required Python packages:
    Install the latest version of Pytorch depending on your CUDA installation:
    eg: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Once you install torch, then run the below command to install other dependencies.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open a web browser and navigate to `http://127.0.0.1:5000/` to access the application. 

### Running Tests

1. Ensure your virtual environment is activated.
2. Run the unit tests from the root directory of the project:
    ```bash
    python -m unittest discover -s test
    ```

### Using Docker

#### Building the Docker Image

1. Build the Docker image:
    ```bash
    docker build -t llm-text-to-node-app .
    ```

#### Running the Application in Docker

1. Run the Docker container:
    ```bash
    docker run -p 5000:5000 llm-text-to-node-app
    ```

2. Open a web browser and navigate to `http://127.0.0.1:5000/` to access the application.

## Project Details

### `app.py`

The main Flask application that handles routing and text classification.

### `models.py`

Contains functions for loading models, performing zero-shot classification, and filtering known nodes.

### `requirements.txt`

Lists the Python dependencies required for the project.

### `Dockerfile`

Defines the Docker image, including the Python environment and dependencies.

### `templates/`

Contains the HTML templates for rendering the web pages.

### `static/`

Contains static files like CSS and JavaScript.

### `test/`

Contains unit tests for the application and models.

- **`App_Unit_Test.py`**: Tests the Flask application's endpoints and ensures proper functionality.
- **`metrics.py`**: Evaluates the model's performance on a set of test cases.
- **`Models_Unit_test.py`**: Unit tests for the model-related functions to ensure they work as expected.
- **`testcases.json`**: Contains 3 test cases for the unit tests.

## Conclusion

This project demonstrates how to build a Flask web application that leverages pre-trained language models for text classification. It includes a complete setup for development, testing, and deployment using Docker. The unit tests ensure that the application functions correctly and provides accurate results.
