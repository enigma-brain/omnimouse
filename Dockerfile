FROM nvcr.io/nvidia/pytorch:25.05-py3
ENV DEBIAN_FRONTEND=noninteractive

# Add GitHub CLI apt repository
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      | tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  wget \
  git \
  gh \
  openssh-client \
  python3 \
  python3-pip \
  python3-dev \
  python3-venv \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libgl1 \
  tree \
  rsync \
  tmux \
  zsh \
  sudo \
  jq \
  less \
  fzf \
  nano \
  vim \
  unzip \
  gnupg2 \
  man-db \
  procps \
  && rm -rf /var/lib/apt/lists/*

# Install git-delta for nicer diffs
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) && \
    wget "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" && \
    rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /src/omnimouse

# Create non-root user matching host UID/GID
# Required: USER_ID, GROUP_ID, USERNAME must be set in .env
ARG USER_ID
ARG GROUP_ID
ARG USERNAME
RUN test -n "${USER_ID}" && test -n "${GROUP_ID}" && test -n "${USERNAME}" \
      || { echo "ERROR: USER_ID, GROUP_ID, and USERNAME must be set in .env. Run:"; \
           echo '  echo -e "USER_ID=$(id -u)\nGROUP_ID=$(id -g)\nUSERNAME=$(whoami)" >> .env'; \
           exit 1; } && \
    groupadd -g ${GROUP_ID} ${USERNAME} 2>/dev/null || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/zsh ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R ${USER_ID}:${GROUP_ID} /src/omnimouse && \
    mkdir -p /home/${USERNAME}/.claude /home/${USERNAME}/.commandhistory && \
    chown ${USER_ID}:${GROUP_ID} /home/${USERNAME}/.claude /home/${USERNAME}/.commandhistory

USER ${USERNAME}

# Set up oh-my-zsh + powerlevel10k
ARG ZSH_IN_DOCKER_VERSION=1.2.0
ENV SHELL=/bin/zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
    -t https://github.com/romkatv/powerlevel10k \
    -p git \
    -p fzf \
    -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
    -a "source /usr/share/doc/fzf/examples/completion.zsh" \
    -a "export HISTFILE=/home/${USERNAME}/.commandhistory/.zsh_history" \
    -a "export SSH_AUTH_SOCK=/ssh-agent" \
    -x

# Bash history persistence + SSH agent socket
RUN echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/home/${USERNAME}/.commandhistory/.bash_history" >> /home/${USERNAME}/.bashrc && \
    echo 'export SSH_AUTH_SOCK=/ssh-agent' >> /home/${USERNAME}/.bashrc && \
    echo 'export SSH_AUTH_SOCK=/ssh-agent' >> /home/${USERNAME}/.profile

# Copy minimal files needed for package install
COPY --chown=${USER_ID}:${GROUP_ID} pyproject.toml uv.lock* .python-version* ./
COPY --chown=${USER_ID}:${GROUP_ID} omnimouse/ ./omnimouse/

# Install everything including local package as the non-root user
RUN uv sync

RUN git config --global --add safe.directory /src/omnimouse/

# Default editor + env
ENV EDITOR=nano
ENV VISUAL=nano
ENV PATH="/src/omnimouse/.venv/bin:/home/${USERNAME}/.local/bin:$PATH"
EXPOSE 8888
