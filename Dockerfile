FROM ghcr.io/darkstar1997/opencv-cuda:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    ninja-build \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cuda-python==12.8
RUN pip install --no-cache-dir torch==2.8 torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir scikit-learn==1.4.2
RUN pip install --no-cache-dir "numpy<2" absl-py attrs flatbuffers "protobuf<5,>=4.25.3" matplotlib
RUN pip install --no-cache-dir "setuptools<82" wheel packaging
RUN pip install --no-cache-dir tqdm==4.66.4
RUN pip install --no-cache-dir pandas==2.2.2
RUN pip install --no-cache-dir Pillow==10.3.0
RUN pip install --no-cache-dir seaborn==0.13.2
RUN pip install --no-cache-dir --no-deps grad-cam ttach

WORKDIR /app

RUN mkdir -p /app/outputs/models /app/outputs/results

CMD ["bash", "/app/runScript.sh"]
