FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system utilities, Python, and Unity standalone headless dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    xvfb \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libvulkan1 \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglu1-mesa \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python requirements
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Make simulator binary and entrypoint executable
RUN chmod +x /app/docker/entrypoint.sh || true
RUN if [ -f "/app/simulator/AutoDRIVE Simulator.x86_64" ]; then chmod +x "/app/simulator/AutoDRIVE Simulator.x86_64"; fi

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "-m", "src.ui.app"]
