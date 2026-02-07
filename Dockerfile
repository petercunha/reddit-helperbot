FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium

# Copy rest of the application
COPY . .

# Run the bot with unbuffered output
CMD ["python", "-u", "main.py"]
