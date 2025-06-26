FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements (if you have a requirements.txt)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose both FastAPI (8080) and Streamlit (8501) ports
EXPOSE 8080 8501

# Start FastAPI (which will also start Streamlit via your startup event)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]