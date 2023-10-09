# Set base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables for user
ENV GROUP_NAME=DHLAB-unit
ENV GROUP_ID=11703

ENV USER_NAME=rusnak
ENV USER_ID=210939

# Install build tools and libraries
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    git  \
    curl  \
    vim  \
    unzip  \
    wget  \
    tmux  \
    screen  \
    wget \
    sudo


RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN sudo DEBIAN_FRONTEND=noninteractive dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
RUN sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get update
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "Debug: USER_ID=$USER_ID, GROUP_ID=$GROUP_ID, USER_NAME=$USER_NAME, GROUP_NAME=$GROUP_NAME"

# Create a group and user
RUN groupadd -g $GROUP_ID $GROUP_NAME
RUN useradd -ms /bin/bash -u $USER_ID -g $GROUP_ID $USER_NAME

# Add new user to sudoers
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add Conda & Java
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
ENV CONDA_PREFIX=/home/${USER_NAME}/.conda
ENV CONDA=/home/${USER_NAME}/.conda/condabin/conda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_PREFIX} && \
    rm miniconda.sh && \
    ${CONDA} config --set auto_activate_base false && \
    ${CONDA} init bash && \
    ${CONDA} create --name myenv python=3.8

ENV PATH="/home/${USER_NAME}/.conda/envs/myenv/bin:$PATH"

RUN /home/${USER_NAME}/.conda/condabin/conda create -n myenv python=3.8 pip

RUN /home/${USER_NAME}/.conda/condabin/conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

RUN /home/${USER_NAME}/.conda/condabin/conda run -n myenv pip install --upgrade pip setuptools wheel
RUN /home/${USER_NAME}/.conda/condabin/conda run -n myenv pip install \
        numpy scipy scikit-learn \
        matplotlib seaborn \
        pillow functorch \
        ninja nvidia-pyindex


RUN /home/${USER_NAME}/.conda/condabin/conda run -n myenv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN git clone --recursive https://github.com/nvlabs/tiny-cuda-nn

RUN cmake tiny-cuda-nn/. -B build
RUN cmake --build build --config RelWithDebInfo -j


RUN /home/${USER_NAME}/.conda/condabin/conda run -n myenv pip install nerfstudio

RUN /home/${USER_NAME}/.conda/condabin/conda run -n myenv pip install genalog==0.1.0 --no-deps

ENV PATH="/home/${USER_NAME}/.conda/envs/myenv/bin:$PATH"
ENV TRANSFORMERS_CACHE="/home/${USER_NAME}/dhlab-data/data/.cache/"

# Set the working directory
WORKDIR /home/$USER_NAME/app

# Copy app directory
COPY . .

# Change ownership of the copied files to the new user and group
RUN chown -R $USER_NAME:$GROUP_NAME /home/$USER_NAME/app

# Switch to the new user
USER $USER_NAME

# Make sure your script is executable
# RUN chmod +x run_one.sh

CMD ["sleep", "infinity"]



