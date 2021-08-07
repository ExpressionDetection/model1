FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

ENV POETRY_VERSION=1.1.7
ENV RPC_PORT=50051
ENV MODEL_ARTIFACT_DIR=/app/model1/models/OuluCASIA
ENV PYTHONUNBUFFERED=1
ENV RELOAD_APP_ON_FILE_CHANGE=true
ENV POETRY_VIRTUALENVS_CREATE=false

# Install prerequisites
RUN apt-get update && \
    apt-get install -y git curl inotify-tools python3.7 python3-pip ffmpeg libsm6 libxext6 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    python3 -m pip install --upgrade pip && \
    curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3 && \
    echo "source $HOME/.poetry/env" > ~/.bashrc

SHELL ["/bin/bash", "-c"]

WORKDIR /app

COPY .ssh /root/.ssh
COPY . /app

# Install dependencies
RUN source ~/.bashrc && poetry lock && poetry install --no-interaction --no-ansi

CMD /app/boot.sh