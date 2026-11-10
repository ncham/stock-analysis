# Use Python base image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the app files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Command to run the app
CMD ["python", "fetch_data.py"]
