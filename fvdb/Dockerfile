FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG MODE=production
RUN echo "Building fVDB container in $MODE mode"

# used for cross-compilation in docker build
ENV FORCE_CUDA=1

WORKDIR /fvdb
COPY . .
RUN  pip install --no-cache-dir -r env/build_requirements.txt

RUN if [ "$MODE" = "production" ]; then \
     MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')  \
     TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" \
     python setup.py install; \
    fi

## Add personal settings
ENV LANG=en_US.UTF-8
# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "Acquire { https::Verify-Peer false }" > /etc/apt/apt.conf.d/99verify-peer.conf \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        git \
        zsh \
        neovim \
        unzip \
        wget \
        rsync \
        vim \
        ninja-build \
        cmake \
        build-essential \
        xauth \
        openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && /root/.fzf/install --all

# Install oh-my-zsh and Install Powerlevel10k and zsh-syntax-highlighting
# hadolint ignore=SC2086
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k" && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

RUN curl -o ~/Miniforge3-$(uname)-$(uname -m).sh -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh && \
    bash ~/Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda

# Install fvdb package
ENV PATH /opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST "6.1;7.0;7.5;8.0;8.6+PTX"
ENV MAX_JOBS=48

RUN conda env create -f env/dev_environment.yml
RUN conda run -n fvdb python setup.py develop

#Install glfw and xvfb for polyscope
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends && \
    apt-get install -y libglfw3-dev && \
    apt-get install -y xvfb
ENV DISPLAY=:99
RUN Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

