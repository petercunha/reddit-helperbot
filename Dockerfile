FROM python:3.12-slim

# Install tini for proper zombie process reaping
RUN apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

# Copy rest of the application
COPY . .

# Use tini as init system to properly reap zombie processes
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the bot with unbuffered output
CMD ["python", "-u", "main.py"]
