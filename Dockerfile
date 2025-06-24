# Use an official Python 3.12 runtime as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies (ignore torch failure)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Install CPU version of PyTorch
RUN pip install torch==2.7.1 --no-cache-dir

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PyTorch and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]