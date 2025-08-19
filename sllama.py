import sys
import subprocess
import shlex
import urllib.request
import json
import os # For file path manipulation and existence checks

def parse_modelfile(filename):
    """
    Parses a Modelfile-like text file and returns a list of arguments
    suitable for passing to llama-cli.

    Supports:
    - FROM <model_id_or_path>: Maps to -m <path> if local file, -hf <id> if Hugging Face ID.
    - PARAMETER <key> <value>: Maps to -<key> <value>
    - SYSTEM \"\"\"<text>\"\"\": Maps to -sys "<text>" (supports multi-line)
    """
    llama_args = []
    in_system_block = False
    system_prompt_lines = []

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()

                # Handle multi-line SYSTEM block
                if in_system_block:
                    if stripped_line == '"""':
                        in_system_block = False
                        # Join lines to form the complete system prompt
                        system_prompt = " ".join(system_prompt_lines).strip()
                        if system_prompt:
                            # Use shlex.quote to ensure the prompt is properly quoted for the shell
                            llama_args.extend(["-sys", shlex.quote(system_prompt)])
                        system_prompt_lines = [] # Reset for potential next system block
                    else:
                        system_prompt_lines.append(line.rstrip('\n')) # Keep newline for multi-line context
                    continue

                if not stripped_line: # Skip empty lines outside system block
                    continue

                # Split the line into command and its value(s)
                # maxsplit=1 ensures that multi-word values (like in PARAMETER) are kept together
                parts = stripped_line.split(maxsplit=1)

                command = parts[0].upper()
                value = parts[1] if len(parts) > 1 else ""

                if command == "FROM":
                    # Check if the value is a path to an existing file
                    if os.path.exists(value) and os.path.isfile(value):
                        # If it's a local file, use the -m (model) flag
                        print(f"Detected local GGUF file: {value}", file=sys.stderr)
                        llama_args.extend(["-m", shlex.quote(value)])
                    else:
                        # Otherwise, assume it's a Hugging Face model ID and use -hf
                        print(f"Assuming Hugging Face model ID: {value}", file=sys.stderr)
                        llama_args.extend(["-hf", shlex.quote(value)])
                elif command == "PARAMETER":
                    param_parts = value.split(maxsplit=1)
                    if len(param_parts) == 2:
                        param_key = param_parts[0]
                        param_value = param_parts[1]
                        llama_args.extend([f"--{param_key}", shlex.quote(param_value)])
                    else:
                        print(f"Warning: Modelfile '{filename}' line {line_num}: Malformed PARAMETER line: '{stripped_line}'", file=sys.stderr)
                elif command == "SYSTEM":
                    if value.startswith('"""'):
                        if len(value) > 3: # If there's text after """ on the same line
                            system_prompt_lines.append(value[3:].strip())
                        in_system_block = True
                    else:
                        # Handle single-line SYSTEM prompts without triple quotes
                        llama_args.extend(["-sys", shlex.quote(value.strip())])
                else:
                    print(f"Warning: Modelfile '{filename}' line {line_num}: Unrecognized command: '{stripped_line}'", file=sys.stderr)

        # If the file ends while still in a system block (missing closing quotes)
        if in_system_block and system_prompt_lines:
             print(f"Warning: Modelfile '{filename}': SYSTEM block not closed with '\"\"\"'. Consuming till EOF.", file=sys.stderr)
             system_prompt = "\n".join(system_prompt_lines).strip() # Use newline for proper multi-line
             if system_prompt:
                llama_args.extend(["-sys", shlex.quote(system_prompt)])

    except FileNotFoundError:
        print(f"Error: Modelfile '{filename}' not found. Please check the path.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing Modelfile '{filename}': {e}", file=sys.stderr)
        sys.exit(1)

    return llama_args

def run_command(executable, args):
    """
    Executes an external command with the given arguments.
    Prints the command being executed and handles common errors.
    """
    command = [executable] + args
    
    print(f"\nExecuting: {' '.join(shlex.quote(arg) for arg in command)}")

    try:
        process = subprocess.Popen(command)
        process.wait() # Wait for the process to complete or be interrupted
        if process.returncode != 0:
            print(f"\nError: Command '{executable}' failed with exit status {process.returncode}.", file=sys.stderr)
            sys.exit(process.returncode)
    except FileNotFoundError:
        print(f"\nError: '{executable}' command not found.", file=sys.stderr)
        print(f"Please ensure '{executable}' is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to run {executable}: {e}", file=sys.stderr)
        sys.exit(1)

def download_from_ollama(model_id):
    """
    Downloads a GGUF model from Ollama's public registry using the OCI Distribution Spec API.
    Prioritizes layers with mediaType 'application/vnd.ollama.image.model'.
    """
    model_name_base, tag = (model_id.split(':', 1) + ['latest'])[:2]
    
    safe_model_name_base = model_name_base.replace('/', '_').replace(':', '-')
    output_filename = f"{safe_model_name_base}-{tag}.gguf"

    if os.path.exists(output_filename):
        print(f"Model '{output_filename}' already exists. Skipping download.", file=sys.stderr)
        return output_filename

    manifest_url = f"https://registry.ollama.ai/v2/library/{model_name_base}/manifests/{tag}"
    print(f"Fetching manifest from: {manifest_url}")

    try:
        req = urllib.request.Request(manifest_url, headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"
        })
        with urllib.request.urlopen(req) as response:
            if response.getcode() != 200:
                print(f"Error: Could not fetch manifest from {manifest_url}. Status code: {response.getcode()}", file=sys.stderr)
                sys.exit(1)
            manifest = json.loads(response.read().decode('utf-8'))

        gguf_digest = None
        
        # Handle manifest lists (image index) first
        if manifest.get('mediaType') in ("application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.index.v1+json"):
            found_manifest_digest = None
            for m in manifest.get('manifests', []):
                if m.get('mediaType') == "application/vnd.ollama.image.manifest.v1+json" or \
                   m.get('mediaType') == "application/vnd.docker.distribution.manifest.v2+json" or \
                   m.get('mediaType') == "application/vnd.oci.image.manifest.v1+json":
                    found_manifest_digest = m.get('digest')
                    break
            
            if found_manifest_digest:
                sub_manifest_url = f"https://registry.ollama.ai/v2/library/{model_name_base}/manifests/{found_manifest_digest}"
                print(f"Fetching specific manifest for model from: {sub_manifest_url}")
                req = urllib.request.Request(sub_manifest_url, headers={
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json"
                })
                with urllib.request.urlopen(req) as sub_response:
                    if sub_response.getcode() != 200:
                        print(f"Error: Could not fetch sub-manifest from {sub_manifest_url}. Status code: {sub_response.getcode()}", file=sys.stderr)
                        sys.exit(1)
                    manifest = json.loads(sub_response.read().decode('utf-8'))
            else:
                print(f"Warning: No specific image manifest found in manifest list for '{model_name_base}:{tag}'. Trying to find digest in top-level manifest config/layers.", file=sys.stderr)

        # *** REFINED LOGIC FOR GGUF DIGEST IDENTIFICATION ***
        # Prioritize layers with specific model media types
        for layer in manifest.get('layers', []):
            if 'digest' in layer and 'mediaType' in layer:
                # This media type typically identifies the GGUF file itself
                if layer['mediaType'] == "application/vnd.ollama.image.model" or \
                   layer['mediaType'].startswith("application/vnd.ollama.image.model.") or \
                   layer['mediaType'] == "application/octet-stream": # General binary blob, often used for GGUF
                    gguf_digest = layer['digest']
                    print(f"Found GGUF digest in layer with mediaType: {layer['mediaType']}", file=sys.stderr)
                    break
        
        # Fallback: Check config digest if no specific model layer was found
        if not gguf_digest:
            config_digest = manifest.get('config', {}).get('digest')
            if config_digest and config_digest.startswith("sha256:"):
                gguf_digest = config_digest
                print("Found GGUF digest in config.", file=sys.stderr)
        # *** END REFINED LOGIC ***

        if not gguf_digest:
            print(f"Error: Could not find GGUF model digest in manifest for '{model_name_base}:{tag}'. No suitable layer or config digest found.", file=sys.stderr)
            print("Please ensure the model ID is correct and its GGUF blob is accessible via the registry API.", file=sys.stderr)
            sys.exit(1)

        download_url = f"https://registry.ollama.ai/v2/library/{model_name_base}/blobs/{gguf_digest}"
        print(f"Downloading GGUF from: {download_url}")
        print(f"Saving to: {output_filename}")

        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\rDownloading: {percent:.1f}% ({readsofar / (1024*1024):.2f}MB / {totalsize / (1024*1024):.2f}MB)"
                sys.stdout.write(s)
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rDownloading: {readsofar / (1024*1024):.2f}MB downloaded...")
                sys.stdout.flush()

        urllib.request.urlretrieve(download_url, output_filename, reporthook=reporthook)
        print("\nDownload complete! 🎉")
        return output_filename

    except urllib.error.HTTPError as e:
        print(f"\nHTTP Error during download: {e.code} - {e.reason}", file=sys.stderr)
        print(f"Error accessing {manifest_url}. This might mean the model or tag isn't directly available via the OCI manifest API or an issue with blob download.", file=sys.stderr)
        print("Possible reasons: Incorrect model/tag, network issues, or a change in Ollama's registry API for blobs.", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\nURL Error during download: {e.reason}", file=sys.stderr)
        print("Check your internet connection or the registry API endpoint.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("\nError: Could not decode JSON response. Invalid manifest or API response from Ollama registry?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during download: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to parse command-line arguments and dispatch to the correct handler.
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python sllama.py modelfile <filename>      - Run llama-cli using instructions from a Modelfile")
        print("  python sllama.py run <gguf_file>          - Run a local GGUF model file")
        print("  python sllama.py run-hug <huggingface_repo> - Run a model directly from Hugging Face")
        print("  python sllama.py serve <gguf_file>        - Start a llama-server instance with a GGUF model")
        print("  python sllama.py dl-from-ollama <model_id> - Download a GGUF model from Ollama's registry (e.g., 'llama3.2:latest' or 'qwen3')")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "modelfile":
        if len(sys.argv) != 3:
            print("Usage: python sllama.py modelfile <filename>", file=sys.stderr)
            sys.exit(1)
        modelfile_path = sys.argv[2]
        llama_args = parse_modelfile(modelfile_path)
        run_command("llama-cli", llama_args)
    elif command == "run":
        if len(sys.argv) != 3:
            print("Usage: python sllama.py run <gguf_file>", file=sys.stderr)
            sys.exit(1)
        gguf_file = sys.argv[2]
        run_command("llama-cli", ["-m", shlex.quote(gguf_file)])
    elif command == "run-hug":
        if len(sys.argv) != 3:
            print("Usage: python sllama.py run-hug <huggingface_repo>", file=sys.stderr)
            sys.exit(1)
        hf_repo = sys.argv[2]
        run_command("llama-cli", ["-hf", shlex.quote(hf_repo)])
    elif command == "serve":
        if len(sys.argv) != 3:
            print("Usage: python sllama.py serve <gguf_file>", file=sys.stderr)
            sys.exit(1)
        gguf_file = sys.argv[2]
        run_command("llama-server", ["-m", shlex.quote(gguf_file), "--port", "11337"])
    elif command == "dl-from-ollama":
        if len(sys.argv) != 3:
            print("Usage: python sllama.py dl-from-ollama <model_id>", file=sys.stderr)
            sys.exit(1)
        model_id = sys.argv[2]
        download_from_ollama(model_id)
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print("Supported commands are: modelfile, run, run-hug, serve, dl-from-ollama", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
