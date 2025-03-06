# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from writing pyc files to disc & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Option 1: Copy and install dependencies from requirements.txt (if available)
# COPY requirements.txt /app/
# RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Option 2: Install dependencies directly.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir flask sqlalchemy psycopg2-binary autogen

# Copy the application code into the container
COPY . /app/

# Expose the port that Flask runs on
EXPOSE 5000

# Run the Flask development server
CMD ["flask", "run"]
