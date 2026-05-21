FROM python:3.11-slim

# Install system dependencies required for headless rendering and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY episode_gen.py .
COPY rl_agents rl_agents

RUN mkdir -p /app/output

CMD ["python", "-u", "episode_gen.py"]