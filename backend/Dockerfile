FROM python:3.12-slim

# Install ffmpeg (ships both ffmpeg and ffprobe) and clean up apt cache
# in the same layer to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first so Docker can cache the pip install layer
# independently of application code changes.
COPY requirements.txt .

# Create a virtual environment and install Python dependencies.
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
