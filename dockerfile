# Use an official Python base image
FROM python:3.11-slim

# Set environment variables for the virtual environment
ENV VENV_PATH="/opt/venv"
ENV PATH="${VENV_PATH}/bin:$PATH"

# Create a virtual environment and install dependencies
RUN python -m venv $VENV_PATH

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache if dependencies don't change
COPY requirements.txt .

# Install dependencies inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY streamlit_app_chromadb.py .

# Expose the port that Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app_chromadb.py", "--server.port=8501", "--server.address=0.0.0.0"]