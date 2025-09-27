# Multi-stage Dockerfile for Online Model Learning Framework
# Stage 1: Base environment with Python and system dependencies
FROM ubuntu:22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Stage 2: Build environment with planners
FROM base AS builder

WORKDIR /opt

# Install Fast Downward
RUN git clone https://github.com/aibasel/downward.git fast-downward && \
    cd fast-downward && \
    python build.py

# Install VAL validator (for plan validation)
RUN git clone https://github.com/KCL-Planning/VAL.git && \
    cd VAL && \
    make && \
    make install

# Stage 3: Development environment
FROM base AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy built planners from builder stage
COPY --from=builder /opt/fast-downward /opt/fast-downward
COPY --from=builder /opt/VAL /opt/VAL

# Set environment variables for planners
ENV FAST_DOWNWARD_PATH=/opt/fast-downward
ENV VAL_PATH=/opt/VAL/bin
ENV PATH="${VAL_PATH}:${PATH}"

# Create workspace directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt requirements-test.txt /workspace/

# Install core dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development/testing dependencies
RUN pip install --no-cache-dir -r requirements-test.txt

# Install linting tools
RUN pip install --no-cache-dir black flake8

# Copy project files
COPY . /workspace/

# Clone external repositories (OLAM and ModelLearner)
WORKDIR /opt
RUN git clone https://github.com/LamannaLeonardo/OLAM.git || true
RUN git clone https://github.com/kcleung/ModelLearner.git || true

# Add external repos to Python path
ENV PYTHONPATH="/opt/OLAM:/opt/ModelLearner/src:${PYTHONPATH}"

# Return to workspace
WORKDIR /workspace

# Create results directory
RUN mkdir -p results

# Stage 4: Testing environment
FROM development AS testing

# Testing dependencies are already installed from requirements-test.txt
# Run tests by default
CMD ["pytest", "-v", "--tb=short", "tests/"]

# Stage 5: Production environment
FROM base AS production

# Copy only necessary files
COPY --from=builder /opt/fast-downward /opt/fast-downward
COPY --from=builder /opt/VAL /opt/VAL

# Set environment variables
ENV FAST_DOWNWARD_PATH=/opt/fast-downward
ENV VAL_PATH=/opt/VAL/bin
ENV PATH="${VAL_PATH}:${PATH}"

WORKDIR /app

# Copy and install only production requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/
COPY benchmarks/ /app/benchmarks/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Clone external repositories
WORKDIR /opt
RUN git clone https://github.com/LamannaLeonardo/OLAM.git || true
RUN git clone https://github.com/kcleung/ModelLearner.git || true

# Add to Python path
ENV PYTHONPATH="/opt/OLAM:/opt/ModelLearner/src:/app:${PYTHONPATH}"

WORKDIR /app

# Default command for production
CMD ["python", "scripts/run_experiments.py"]