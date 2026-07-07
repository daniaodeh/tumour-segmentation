FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workdir

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenslide-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install Rust and compile the tile_with_overlap binary
ENV CARGO_HOME="/usr/local/.cargo" RUSTUP_HOME="/usr/local/.rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/usr/local/.cargo/bin:${PATH}"
RUN cargo build --manifest-path=preprocess/tile_with_overlap/Cargo.toml --release

# Install python packages
RUN python -m pip install --upgrade pip && \
    python -m pip uninstall -y opencv_python opencv-python && \
    python -m pip install --no-cache-dir \
    pandas \
    opencv_python_headless \
    albumentations==1.1.0 \
    timm==0.4.12 \
    segmentation-models-pytorch==0.2.1 \
    torchinfo==1.6.3 \
    toml==0.10.2 \
    polars \
    scikit-image \
    scipy \
    tqdm \
    openslide-python

# Verify GPU compatibility at build time (optional but useful)
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"