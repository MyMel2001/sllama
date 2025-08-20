# sllama: A Llama.cpp CLI Frontend

`sllama` is a lightweight command-line interface (CLI) frontend for `llama.cpp`'s executable tools (`llama-cli` and `llama-server`). It aims to provide an **Ollama-like user experience** for managing and interacting with GGUF models, especially tailored for environments where Ollama might not be natively supported, such as **Windows 7**.

## Why `sllama`?

This project was created out of a personal need for an Ollama-like interface that could run on older operating systems, specifically Windows 7, which mainstream Ollama does not officially support. `sllama` bridges this gap by providing a familiar command structure to interact with the robust `llama.cpp` tools without relying on the full Ollama ecosystem.

## Features

* **Modelfile Support**: Define model configurations (FROM, PARAMETER, SYSTEM prompts) in simple text files, similar to Ollama's Modelfiles.
* **Local GGUF Execution**: Easily run local `.gguf` model files directly.
* **Hugging Face Integration**: Run models directly from Hugging Face repositories.
* **Built-in Server**: Launch a `llama.cpp` HTTP server (`llama-server`) for API access to your models.
* **Ollama Registry Download**: Download GGUF models directly from Ollama's public model registry without needing the Ollama application.
* **No External Python Dependencies**: Built purely with Python's standard library, ensuring minimal overhead and broad compatibility.

## Installation

To install `sllama`, follow these steps:

1.  **Prerequisites:**
    * Python 3.7 or newer.
    * `llama.cpp` compiled executables (`llama-cli` and `llama-server`) must be available in your system's PATH. You can find instructions on how to compile `llama.cpp` [here](https://github.com/ggerganov/llama.cpp) - or you can download a pre-compiled "Vulkan" copy. This tends to work best for Windows 7.

2.  **Download the project:**
    Clone this repository or download the `sllama.py` file and the `setup.py` file.
    ```bash
    git clone https://git.nodemixaholic.com/sparky/sllama.git
    cd sllama
    ```

3.  **Install `build` module (if not already installed):**
    ```bash
    pip install build
    ```

4.  **Build and Install:**
    Navigate to the `sllama_cli` directory (where `sllama.py` and `setup.py` are located) and run:
    ```bash
    python -m build
    pip install ./dist/sllama_cli-0.1.0-py3-none-any.whl # Adjust filename to what 'python -m build' outputs
    ```
    Alternatively, for development, you can install in "editable" mode:
    ```bash
    pip install -e .
    ```

## Usage

Once installed, you can use the `sllama` command from your terminal.

### 1. Running a Modelfile (`modelfile`)

Create a text file (e.g., `my_model_config.txt`) with your model instructions (also found in `linus-test.txt` if you obtained the source.):

```
FROM Qwen/Qwen3-14B-GGUF
PARAMETER temp 0.75
SYSTEM """
You are a helpful assistant named Linus.
You like to help the user. You should always be friendly and helpful.
"""
```

If the `FROM` instruction points to an existing local GGUF file, it will automatically use the `-m` flag; otherwise, it defaults to `-hf` for Hugging Face models.

To run it:

```
sllama modelfile my_model_config.txt
```

2. Running a Local GGUF File (run)
To run a GGUF file you already have downloaded (e.g., my_local_model.gguf):

```sllama run my_local_model.gguf```

3. Running a Hugging Face Model (run-hug)
To directly download and run a model from Hugging Face:

```sllama run-hug mistralai/Mixtral-8x7B-Instruct-v0.1```

4. Starting a Model Server (serve)
To start an HTTP API server for a GGUF model (runs on port 11337 to avoid conflicts with Ollama):

```sllama serve my_api_model.gguf```

This will run llama-server -m my_api_model.gguf --port 11337.

5. Downloading from Ollama Registry (dl-from-ollama)
Download GGUF models directly from Ollama's public registry. If no tag is specified, it defaults to latest. The file will be saved in your current directory.

```
sllama dl-from-ollama llama3.2:latest
# Or with default tag:
sllama dl-from-ollama llama2
```


## License

This project is open-source and distributed under the MIT License. See the LICENSE file for details.