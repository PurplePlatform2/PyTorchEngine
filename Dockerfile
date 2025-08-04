# Use the official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (change if needed)
EXPOSE 8000

# Run the application (change main.py if your entry is different)
CMD ["python", "main.py"]
