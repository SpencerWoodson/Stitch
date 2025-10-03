# Stitch

## Required components

### Installation Guide

This project uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with CUDA acceleration.  
Follow the steps below to set up your environment.

---

#### 1. System Requirements

- Linux (tested on Arch Linux)
- NVIDIA GPU (tested on RTX 4060)
- NVIDIA drivers (>= 550 for CUDA 13)
- CUDA Toolkit (installed via `pacman`)
- Python 3.10+ (tested on 3.13)
- cmake and gcc for building native code

Note: If installing without CUDA acceleration, you don't need cuda or nvidia gpu/drivers

---

#### 2. Install System Packages

On Arch Linux:
```bash
sudo pacman -S --needed cuda gcc cmake python python-pip
```
Note: Remove cuda gcc and cmake if you don't want to use GPU Acceleration

Check that nvcc is available in your path:

```bash
nvcc --version
```

If nvcc is not found, add this to your shell config (~/.bashrc or ~/.zshrc) or run in your terminal directly 

```bash
export PATH=/opt/cuda/bin:$PATH
export CUDAToolkit_Root=/opt/cuda
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

If you updated your shell config:

```bash 
source ~/.bashrc
```
#### 3. Download LLM models
This project downloaded its models from HuggingFace using the following links

Fast(recommended Q6_K_L):
https://huggingface.co/bartowski/Ministral-8B-Instruct-2410-GGUF

Deep (recommended Q4_K_M):
https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF

You can use whatever models you want and edit the config file as necessary

#### 4. Set up a Python environment
Create a virtual environment
```bash 
python -m venv llama-env
source llama-env/bin/activate
pip install --upgrade pip
```

Install requirements for this repo:
```bash
pip install -r requirements.txt
```

#### 5. (OPTIONAL) Build llama-cpp-python with CUDA support
Automatically, llama-cpp-python is CPU-only. If you want to enable GPU-Acceleration, 
```bash
CMAKE_ARGS="-GGML_CUDA=on" pip install --force-reinstall --no-cache-dir llama-cpp-python
```

#### 6. Configure config.yaml
Important configs to change:
- VAULT_DIR: initially set to be in your home directory and named Vault
- models (fast and deep): initially set to be in a models folder within the project directory. Make sure to update with whichever models you chose

#### 7. (OPTIONAL) Create a bash script to run directly from your terminal

Create the bash script (recommended location /usr/local/bin)
```bash
#!/bin/bash

STITCH_PATH="$HOME/Stitch"
VENV_PYTHON="$STITCH_PATH/llama-env/bin/python"
MAIN_SCRIPT="$STITCH_PATH/scripts/main.py"

# Pass all arguments to main.py
"$VENV_PYTHON" "$MAIN_SCRIPT" "$@"
```
Weirdly I can't find where I added it to my path but it should be here in an ideal world. Then edit your shell config file:
```bash
nano ~/.bashrc
```

Then put 
```bash 
export PATH="$HOME/stitch:$PATH"
``` 

And reload your shell 
```bash
source ~/.bashrc
```

#### 7. Run the project
Once you've installed everything, you can activate your environment and run: 
```bash
source llama-env/bin/activate
python main.py
```


