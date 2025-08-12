# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    gnupg \
    wget \
    git \
    vim \
    sudo \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    software-properties-common \
    ca-certificates \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglx-mesa0 \
    libgles2-mesa \
    libxcb1 \
    # WSLg testing tools
    mesa-utils \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo Harmonic
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update && \
    apt-get install -y gz-harmonic && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/pid_rl
ENV HOME=/home/pid_rl

# Install ArduPilot dependencies
RUN apt-get update && apt-get install -y \
    python3-numpy \
    python3-empy \
    python3-toml \
    python3-future \
    python3-lxml \
    libxml2-dev \
    libxslt-dev \
    gawk \
    libtool \
    libglib2.0-dev \
    autotools-dev \
    automake \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies for ArduPilot-Gazebo plugin
RUN apt-get update && apt-get install -y \
    libgz-sim8-dev \
    rapidjson-dev \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    && rm -rf /var/lib/apt/lists/*

# Clone ArduPilot
RUN git clone https://github.com/ArduPilot/ardupilot.git && \
    cd ardupilot && \
    git submodule update --init --recursive

# Create non-root user for running scripts
RUN useradd -ms /bin/bash ardupilot && \
    usermod -aG sudo ardupilot && \
    echo "ardupilot ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Fix ownership for pip installs
RUN chown -R ardupilot:ardupilot /home/pid_rl

# Run install-prereqs as non-root
USER ardupilot
WORKDIR /home/pid_rl/ardupilot
RUN USER=ardupilot ./Tools/environment_install/install-prereqs-ubuntu.sh -y && \
    . ~/.profile

# Switch back to root for remaining steps
USER root
WORKDIR /home/pid_rl

# Clone and build ArduPilot-Gazebo plugin
RUN git clone https://github.com/ArduPilot/ardupilot_gazebo.git && \
    cd ardupilot_gazebo && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
    make -j$(nproc)

# Set Gazebo environment variables
ENV GZ_VERSION=harmonic
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=/home/pid_rl/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
ENV GZ_SIM_RESOURCE_PATH=/home/pid_rl/ardupilot_gazebo/models:/home/pid_rl/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}

# Set up ArduPilot environment
ENV PATH=/home/pid_rl/ardupilot/Tools/autotest:${PATH}
ENV PYTHONPATH=/home/pid_rl/ardupilot/Tools/autotest:${PYTHONPATH}

# WSLg GPU library path (essential for GPU acceleration)
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install numpy matplotlib==3.10.0 scipy pandas mavsdk gymnasium mavproxy==1.8.71 protobuf==5.29.0 

# Build ArduPilot SITL targets (copter, plane, rover)
WORKDIR /home/pid_rl/ardupilot
RUN  git config --global --add safe.directory /home/pid_rl/ardupilot
RUN ./waf configure --board sitl && \
    ./waf copter -j$(nproc)

RUN echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc

CMD ["/bin/bash"]