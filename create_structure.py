import os

# Define the directory and file structure
structure = {
    "templates": ["layout.html", "index.html", "result.html"],
    "static": ["styles.css"],
    "": ["app.py", "models.py"]
}

# Create the directory structure and files
for folder, files in structure.items():
    if folder:
        os.makedirs(folder, exist_ok=True)
    for file in files:
        with open(os.path.join(folder, file), 'w') as f:
            # Create empty files
            pass

print("Directory structure and files created successfully.")
