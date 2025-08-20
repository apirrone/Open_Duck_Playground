FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Set PATH and LD_PRELOAD so that uv and the preloading are set system-wide.
ENV PATH="/root/.local/bin:${PATH}"
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

# Install system tools, enable Universe repository, and install X11/GL dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      curl \
      git && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libx11-6 \
      libgl1-mesa-glx \
      libglx-mesa0 && \
    # Now add the deadsnakes PPA for Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 \
      python3.11-distutils \
      python3.11-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    python -m pip install --upgrade pip

WORKDIR /open_duck
# Copy the entire repository; use a proper .dockerignore to exclude unneeded files.
COPY . .

# Install uv, then ensure the uv lockfile exists (create if not) and sync the environment.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    if [ ! -f uv.lock ]; then uv lock; fi && \
    uv sync --frozen

CMD ["bash"]
