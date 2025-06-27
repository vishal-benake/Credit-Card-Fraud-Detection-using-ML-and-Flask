# Use a lightweight base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "application.py"]