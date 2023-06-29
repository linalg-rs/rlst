FROM mcr.microsoft.com/devcontainers/rust:bullseye

RUN apt-get update \
  && apt-get upgrade -y \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get -y install --no-install-recommends build-essential libssl-dev pkg-config libmpich-dev libclang-dev \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*
