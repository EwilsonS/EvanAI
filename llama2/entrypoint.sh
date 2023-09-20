#!/bin/bash

# Set the default value if LLAMA_MODEL is unset or empty
LLAMA_DOWNLOAD_URL=${LLAMA_DOWNLOAD_URL:-https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main}
LLAMA_MODEL=${LLAMA_MODEL:-llama-2-7b.Q2_K.gguf}
PIP_INSTALL_OPTIONS=${PIP_INSTALL_OPTIONS:-"--index-url https://pypi.org/simple"}
read -a PIP_INSTALL_ARGS <<< $PIP_INSTALL_OPTIONS

# Directory to check/store the model
MODEL_DIR="/tmp/models"
MODEL_PATH="${MODEL_DIR}/${LLAMA_MODEL}"

# Install requirements
VENV_DIR=$(mktemp -d)
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
echo "PIP_INSTALL_OPTIONS: ${PIP_INSTALL_ARGS[@]}"
pip3 install ${PIP_INSTALL_ARGS[@]} --upgrade pip || exit 1
pip3 install ${PIP_INSTALL_ARGS[@]} llama-cpp-python[server] || exit 1

# Create the directory if it doesn't exist
mkdir -p "${MODEL_DIR}"

# Check if the model file already exists. If not, download it.
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Downloading ${LLAMA_DOWNLOAD_URL}/${LLAMA_MODEL} model..."
    curl -L "${LLAMA_DOWNLOAD_URL}/${LLAMA_MODEL}" -o "${MODEL_PATH}"
fi

# Run fastapi server
ls -lah $MODEL_PATH
python3 -m llama_cpp.server --model $MODEL_PATH --host 0.0.0.0 --port 8000
