# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Expose the port the app runs on
EXPOSE 5000

# Define environment variable to avoid byte-compiled .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Define environment variable to buffer stdout and stderr
ENV PYTHONUNBUFFERED 1

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the Flask app
CMD ["python", "app.py"]
