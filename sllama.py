import sys
import subprocess
import shlex
import urllib.request
import json
import os
import socket
import threading
import http.server
from urllib.parse import urlparse
import requests
import time
from subprocess import DEVNULL # Import DEVNULL from subprocess for cross-platform compatibility

# Dictionary to store information about running llama-server processes
# Keys are model names (e.g., "deepseek-r1.gguf"), values are dictionaries with 'port' and 'process'
running_models = {}

def find_free_port():
    """Finds a free port on the system by binding to a random ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # Bind to port 0 to let the OS choose a free port
        return s.getsockname()[1] # Return the chosen port number

def run_command_in_background(executable, args, name, port):
    """
    Executes an external command (like llama-server) in the background.
    It stores the process object in 'running_models' for later management.
    Stdout and stderr are redirected to DEVNULL to prevent terminal output clutter.
    """
    command = [executable] + args
    print(f"\nExecuting '{name}' in background: {' '.join(shlex.quote(arg) for arg in command)}")
    try:
        # Popen runs the command asynchronously.
        # stdout and stderr are redirected to subprocess.DEVNULL for clean output.
        process = subprocess.Popen(command, stdout=DEVNULL, stderr=DEVNULL)
        running_models[name] = {'port': port, 'process': process}
        print(f"Server for '{name}' started on port {port} with PID {process.pid}.")
    except FileNotFoundError:
        print(f"\nError: '{executable}' command not found.", file=sys.stderr)
        print(f"Please ensure '{executable}' is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to run {executable}: {e}", file=sys.stderr)
        sys.exit(1)

def is_port_in_use(port):
    """Checks if a given TCP port is currently in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port)) # Try to bind to the port
            return False # If successful, port is free
        except OSError:
            return True # If binding fails, port is in use

class LlamaRouter(http.server.BaseHTTPRequestHandler):
    """
    A reverse proxy that routes incoming HTTP requests to the correct
    llama-server instance, supporting both OpenAI-like API calls
    (model in JSON body or specific /v1/models paths) and custom routing
    (model name as first path segment).
    """

    def _forward_request(self, method, model_name, forwarded_path, body=None):
        """
        Helper method to forward an HTTP request to a target URL with
        exponential backoff for retries.
        model_name: The identified name of the model (e.g., "deepseek-r1.gguf")
        forwarded_path: The path to send to the backend llama-server (e.g., "/v1/chat/completions")
        """
        if model_name not in running_models:
            self.send_error(404, f"Model '{model_name}' not found or not running.")
            return

        target_port = running_models[model_name]['port']
        # Construct the target URL using the backend's port and the rewritten path
        target_url = f"http://localhost:{target_port}{forwarded_path}"
        
        # Preserve original query parameters
        parsed_original_path = urlparse(self.path)
        if parsed_original_path.query:
            target_url += f"?{parsed_original_path.query}"

        print(f"Routing {method} request for model '{model_name}' to backend: {target_url}")

        retries = 3
        backoff_factor = 0.5 # Initial delay in seconds

        for i in range(retries):
            try:
                # Prepare headers for forwarding.
                # Important: Set 'Host' header to the actual target server's host:port
                # so the backend server receives the correct Host.
                headers = dict(self.headers)
                headers['Host'] = f"localhost:{target_port}"
                
                # Make the request to the backend llama-server
                response = requests.request(
                    method,
                    target_url,
                    headers=headers,
                    data=body,
                    stream=True, # Use stream=True to handle large responses efficiently
                    timeout=600 # Long timeout for LLM inference
                )
                break # Request successful, exit retry loop
            except requests.exceptions.ConnectionError as e:
                # Handle connection errors (e.g., backend server not ready)
                print(f"Connection error to {target_url}: {e}. Retrying in {backoff_factor * (2 ** i):.1f} seconds...", file=sys.stderr)
                time.sleep(backoff_factor * (2 ** i)) # Exponential backoff
            except requests.exceptions.Timeout as e:
                # Handle timeouts
                print(f"Timeout connecting to {target_url}: {e}. Retrying...", file=sys.stderr)
                time.sleep(backoff_factor * (2 ** i)) # Exponential backoff
        else:
            # All retries failed
            self.send_error(504, f"Failed to connect to model server after {retries} retries.")
            return

        # Forward the response from the backend server back to the original client
        self.send_response(response.status_code)
        for key, value in response.headers.items():
            # Avoid forwarding hop-by-hop headers that are handled by proxies themselves
            if key.lower() not in ['content-encoding', 'transfer-encoding', 'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'upgrade']:
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response.content) # Write the response content (bytes)

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path_segments = parsed_path.path.strip('/').split('/')

        # Case 1: OpenAI-standard /v1/models endpoint
        # Example: GET http://localhost:11337/v1/models
        # Example: GET http://localhost:11337/v1/models/model_id
        if path_segments and path_segments[0] == "v1" and len(path_segments) >= 2 and path_segments[1] == "models":
            # If it's /v1/models or /v1/models/ (list models)
            if len(path_segments) == 2 or (len(path_segments) == 3 and not path_segments[2]):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                models_data = []
                for name, info in running_models.items():
                    if info['process'].poll() is None: # Check if the backend process is still running
                        models_data.append({
                            "id": name,
                            "object": "model",
                            "created": int(time.time()), # Mock creation time
                            "owned_by": "local",
                            "permission": [
                                {"id": f"model-perm-{name}", "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": False, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}
                            ],
                            "root": name,
                            "parent": None
                        })
                
                response_payload = {
                    "object": "list",
                    "data": models_data
                }
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                return
            # If it's /v1/models/<model_id>
            elif len(path_segments) >= 3:
                model_id_from_path = path_segments[2]
                if model_id_from_path in running_models:
                    # Forward exactly '/v1/models/<model_id>' to the backend
                    forwarded_path = f"/v1/models/{model_id_from_path}"
                    self._forward_request(method='GET', model_name=model_id_from_path, forwarded_path=forwarded_path)
                    return
                else:
                    self.send_error(404, f"Model '{model_id_from_path}' not found or not running.")
                    return
        
        # Case 2: Custom Routing - model name as first path segment
        # Example: GET http://localhost:11337/deepseek-r1.gguf/v1/chat/completions
        if path_segments and path_segments[0] in running_models:
            model_name_from_path = path_segments[0]
            # Rewrite path: remove the model name segment for the backend server
            forwarded_path = '/' + '/'.join(path_segments[1:])
            # Ensure the forwarded path starts with /
            if not forwarded_path.startswith('/'):
                forwarded_path = '/' + forwarded_path
            
            self._forward_request(method='GET', model_name=model_name_from_path, forwarded_path=forwarded_path)
            return

        # If neither standard OpenAI nor custom path routing matches
        self.send_error(404, "Unsupported GET endpoint or model not found.")


    def do_POST(self):
        parsed_path = urlparse(self.path)
        path_segments = parsed_path.path.strip('/').split('/')
        
        body = None
        request_payload = {}
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                request_payload = json.loads(body.decode('utf-8'))
        except (ValueError, json.JSONDecodeError) as e:
            self.send_error(400, f"Invalid JSON in request body: {e}")
            return
        except Exception as e:
            print(f"Warning: Could not read or decode POST body: {e}", file=sys.stderr)
            pass 

        # Case 1: OpenAI-standard POST endpoints (e.g., /v1/chat/completions, /v1/completions)
        # Model name is expected in the JSON body
        # Example: POST http://localhost:11337/v1/chat/completions (model in body)
        if path_segments and path_segments[0] == "v1" and len(path_segments) >= 2 and \
           path_segments[1] in ["chat", "completions"]: # Covers /v1/chat/completions and /v1/completions
            
            model_name_from_body = request_payload.get("model")

            if not model_name_from_body:
                self.send_error(400, "For /v1/chat/completions or /v1/completions, the 'model' field is required in the JSON body.")
                return

            if model_name_from_body not in running_models:
                self.send_error(404, f"Model '{model_name_from_body}' not found or not running. "
                                     f"Available models: {', '.join(running_models.keys())}")
                return

            # Forward the exact OpenAI-standard path to the backend
            forwarded_path = parsed_path.path # e.g., /v1/chat/completions
            self._forward_request(method='POST', model_name=model_name_from_body, forwarded_path=forwarded_path, body=body)
            return
        
        # Case 2: Custom Routing - model name as first path segment for POST requests
        # Example: POST http://localhost:11337/deepseek-r1.gguf/v1/chat/completions
        if path_segments and path_segments[0] in running_models:
            model_name_from_path = path_segments[0]
            # Rewrite path: remove the model name segment for the backend server
            forwarded_path = '/' + '/'.join(path_segments[1:])
            # Ensure the forwarded path starts with /
            if not forwarded_path.startswith('/'):
                forwarded_path = '/' + forwarded_path
            
            self._forward_request(method='POST', model_name=model_name_from_path, forwarded_path=forwarded_path, body=body)
            return

        # If neither standard OpenAI nor custom path routing matches
        self.send_error(404, "Unsupported POST endpoint or model not specified/found.")


def run_router():
    """Starts the auto-router on the specified fixed port (11337)."""
    router_port = 11337
    if is_port_in_use(router_port):
        print(f"Router is already running on port {router_port}. Skipping starting a new one.")
        return

    print(f"Starting auto-router on port {router_port}...")
    server_address = ('0.0.0.0', router_port)
    # Use ThreadingHTTPServer for better concurrency when handling multiple client requests
    httpd = http.server.ThreadingHTTPServer(server_address, LlamaRouter)
    router_thread = threading.Thread(target=httpd.serve_forever)
    router_thread.daemon = True # Allows the main thread to exit, which will also terminate this thread
    router_thread.start()
    print(f"Auto-router started on http://localhost:{router_port}. Press Ctrl+C to stop all services.")

# --- Original functions (modified only for `run_command_in_background` usage) ---

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

                if in_system_block:
                    if stripped_line == '"""':
                        in_system_block = False
                        system_prompt = " ".join(system_prompt_lines).strip()
                        if system_prompt:
                            llama_args.extend(["-sys", shlex.quote(system_prompt)])
                        system_prompt_lines = []
                    else:
                        system_prompt_lines.append(line.rstrip('\n'))
                    continue

                if not stripped_line:
                    continue

                parts = stripped_line.split(maxsplit=1)

                command = parts[0].upper()
                value = parts[1] if len(parts) > 1 else ""

                if command == "FROM":
                    if os.path.exists(value) and os.path.isfile(value):
                        print(f"Detected local GGUF file: {value}", file=sys.stderr)
                        llama_args.extend(["-m", shlex.quote(value)])
                    else:
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
                        if len(value) > 3:
                            system_prompt_lines.append(value[3:].strip())
                        in_system_block = True
                    else:
                        llama_args.extend(["-sys", shlex.quote(value.strip())])
                else:
                    print(f"Warning: Modelfile '{filename}' line {line_num}: Unrecognized command: '{stripped_line}'", file=sys.stderr)

            if in_system_block and system_prompt_lines:
                print(f"Warning: Modelfile '{filename}': SYSTEM block not closed with '\"\"\"'. Consuming till EOF.", file=sys.stderr)
                system_prompt = "\n".join(system_prompt_lines).strip()
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
    This function is for synchronous, blocking commands (e.g., llama-cli).
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
        print("\nDownload complete! �")
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
    The 'serve' command now exclusively uses the auto-router with random ports for models.
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python sllama.py modelfile <filename>      - Run llama-cli using instructions from a Modelfile")
        print("  python sllama.py run <gguf_file>          - Run a local GGUF model file")
        print("  python sllama.py run-hug <huggingface_repo> - Run a model directly from Hugging Face")
        print("  python sllama.py serve <model_name>=<gguf_file> [<model_name>=<gguf_file>...]")
        print("     - Starts one or more llama-server instances on random ports, with an auto-router on port 11337.")
        print("  python sllama.py dl-from-ollama <model_id> - Download a GGUF model from Ollama's registry (e.g., 'llama3.2:latest' or 'qwen3')")
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
    elif command == "serve": # This is now the unified 'serve' command
        if len(sys.argv) < 3:
            print("Usage: python sllama.py serve <model_name>=<gguf_file> [<model_name>=<gguf_file>...]", file=sys.stderr)
            sys.exit(1)
        
        # Start the router first
        run_router()

        # Start each model server on a random port
        for model_arg in sys.argv[2:]:
            if '=' not in model_arg:
                print(f"Invalid argument format: '{model_arg}'. Expected <model_name>=<gguf_file>", file=sys.stderr)
                continue
            model_name, gguf_file = model_arg.split('=', 1)
            
            if not os.path.exists(gguf_file):
                print(f"Error: Model file '{gguf_file}' not found. Skipping.", file=sys.stderr)
                continue
            
            # Find a free port and start the server
            port = find_free_port()
            run_command_in_background(
                "llama-server", 
                ["-m", shlex.quote(gguf_file), "--port", str(port), "--host", "localhost"], # Explicitly bind to localhost
                model_name,
                port
            )
        
        # Keep the main thread alive so the background servers and router can run
        try:
            # Using Event().wait() is often better for actual applications, but a simple loop
            # with sleep is sufficient to keep the main thread alive for this purpose.
            while True:
                time.sleep(1) 
        except KeyboardInterrupt:
            print("\nShutting down all servers...")
            # Terminate all running processes
            for info in running_models.values():
                if info['process'].poll() is None: # Check if still running before trying to terminate
                    info['process'].terminate()
            print("All services stopped. Goodbye! 👋")
            sys.exit(0)
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