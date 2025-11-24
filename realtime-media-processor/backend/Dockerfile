# Use the official Python image as a base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# 0.0.0.0 makes the server accessible from outside the container
# --host 0.0.0.0 --port 8000 are explicitly set for clarity, though EXPOSE often handles this
# --workers 1 for simplicity, can be adjusted for production needs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
