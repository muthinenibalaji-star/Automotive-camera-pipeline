# Automotive Camera Pipeline - Production Container
# Base: NVIDIA CUDA for GPU-accelerated inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    ffmpeg \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install MMDetection and dependencies
RUN pip install openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.0" && \
    mim install "mmdet>=3.0.0"

# Install core perception dependencies
RUN pip install \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    scipy==1.11.3 \
    pandas==2.1.1 \
    pillow==10.0.1 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    requests==2.31.0

# Install tracking and state estimation dependencies
RUN pip install \
    filterpy==1.4.5 \
    scikit-learn==1.3.1 \
    scikit-image==0.21.0

# Install ByteTrack (tracking)
RUN pip install git+https://github.com/ifzhang/ByteTrack.git

# Install additional utilities
RUN pip install \
    loguru==0.7.2 \
    jsonschema==4.19.1 \
    matplotlib==3.8.0 \
    seaborn==0.13.0

# Create necessary directories
RUN mkdir -p /app/src \
    /app/configs \
    /app/models \
    /app/data/input \
    /app/data/output \
    /app/logs \
    /app/tests

# Copy application code (will be mounted in development)
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for potential API/monitoring
EXPOSE 8080

# Default command (can be overridden)
CMD ["python3", "-m", "src.main"]
