FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY data/ /app/data/

# Make directory structure
RUN mkdir -p /app/input /app/output /app/models /app/PDFs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create entry point script
COPY run_container.sh /app/
RUN chmod +x /app/run_container.sh

# Set the entry point
ENTRYPOINT ["/app/run_container.sh"] 