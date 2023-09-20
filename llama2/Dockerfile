FROM debian:latest

# Install packages
RUN apt update && apt install -y \
    python3 \
    python3-full \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    gcc \
    cmake make curl

# Install Python packages
WORKDIR /app

# entrypoint.sh downloads the model and starts the server
COPY entrypoint.sh ./
EXPOSE 8000
ENTRYPOINT [ "/app/entrypoint.sh" ]

HEALTHCHECK --interval=1m --timeout=3s --retries=3 CMD curl -f http://localhost:8000/docs || exit 1
