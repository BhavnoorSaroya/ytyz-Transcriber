########################################
# Base image with CUDA repo
########################################
FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND=noninteractive

# Install CUDA repo key
RUN apt update -q && apt install -y ca-certificates wget && \
    wget -qO /cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /cuda-keyring.deb && apt update -q


########################################
# Builder: install Python + pip packages
########################################
FROM base as builder

RUN apt update && apt install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip build-essential git ffmpeg

WORKDIR /app
COPY dev.txt .

# Install Python deps globally (simplest model)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r dev.txt


########################################
# Runtime: CUDA + Python runtime + app
########################################
FROM base as runtime

# Install minimal CUDA runtime libs needed for PyTorch + WhisperX
RUN apt update && apt install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg \
    libcudnn8 libcublas-12-2 cuda-nvcc-12-2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages \
                    /usr/local/lib/python3.11/dist-packages

# Copy app source code
COPY . .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r dev.txt

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

