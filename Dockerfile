FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies and purge unnecessary CUDA static libraries
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get purge -y "*static*" && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Set environment variables for compilation
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV FLASH_ATTENTION_SKIP_CUDA_CHECK=1
ENV MAX_JOBS=4

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y transformers accelerate peft bitsandbytes && \
    pip cache purge && \
    pip install --no-cache-dir packaging ninja && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir "flash-attn==2.6.3" --no-build-isolation && \
    rm -rf /root/.cache/pip

# Copy source code and entrypoint
COPY src/ /opt/ml/code/src/
COPY entrypoint.sh /opt/ml/code/

RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]
