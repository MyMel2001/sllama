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

# Dictionary to store information about models.
# Initially: model_name: {'gguf_path': 'path/to/gguf'}
# After activation: model_name: {'gguf_path': 'path/to/gguf', 'port': <port_num>, 'process': <Popen_object>}
registered_models = {}

def find_free_port():
    """Finds a free port on the system by binding to a random ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # Bind to port 0 to let the OS choose a free port
        return s.getsockname()[1] # Return the chosen port number

def run_llama_server_in_background(gguf_path, model_name, port):
    """
    Launches a llama-server process in the background.
    Stdout and stderr are redirected to DEVNULL to prevent terminal output clutter.
    Returns the Popen object for the process.
    """
    executable = "llama-server"
    args = ["-m", shlex.quote(gguf_path), "--port", str(port), "--host", "localhost"]
    command = [executable] + args
    print(f"\nActivating model '{model_name}': Executing '{executable}' in background on port {port}: {' '.join(shlex.quote(arg) for arg in command)}", file=sys.stderr)
    try:
        # stdout and stderr are redirected to DEVNULL for clean output.
        process = subprocess.Popen(command, stdout=DEVNULL, stderr=DEVNULL)
        return process
    except FileNotFoundError:
        print(f"\nError: '{executable}' command not found.", file=sys.stderr)
        print(f"Please ensure '{executable}' is installed and in your system's PATH.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to run {executable} for {model_name}: {e}", file=sys.stderr)
        return None

def wait_for_server_ready(port, model_name, timeout=600): # Default timeout 10 minutes
    """
    Waits for the llama-server on the given port to become truly ready
    by attempting a dummy chat completion request.
    Returns True if ready, False otherwise.
    """
    start_time = time.time()
    
    # First, a basic ping to ensure the server process is listening
    ping_url = f"http://localhost:{port}/v1/models"
    print(f"Waiting for '{model_name}' server process to be listening on {ping_url}...", file=sys.stderr)
    while time.time() - start_time < timeout:
        try:
            response = requests.get(ping_url, timeout=5)
            if response.status_code == 200:
                print(f"'{model_name}' server is listening. Now checking for model readiness...", file=sys.stderr)
                break # Server is listening, proceed to model readiness check
        except requests.exceptions.ConnectionError:
            pass # Server not yet listening
        except requests.exceptions.Timeout:
            pass # Request timed out
        except Exception as e:
            print(f"Unexpected error during initial ping for {model_name}: {e}", file=sys.stderr)
        time.sleep(1)
    else:
        print(f"Error: '{model_name}' server process on port {port} did not start listening within {timeout} seconds.", file=sys.stderr)
        return False

    # Second, attempt a dummy chat completion request to verify model loading
    chat_url = f"http://localhost:{port}/v1/chat/completions"
    dummy_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.1, # Use a low temperature for more deterministic/fast response
        "max_tokens": 1 # Request minimal output
    }
    headers = {"Content-Type": "application/json"}

    print(f"Sending dummy request to '{model_name}' on {chat_url} to check model loading...", file=sys.stderr)
    while time.time() - start_time < timeout:
        try:
            response = requests.post(chat_url, json=dummy_payload, headers=headers, timeout=3600) # Increased single-request timeout

            # Check for success: 200 OK and not containing "loading model" explicitly
            response_text = response.text.lower()
            if response.status_code == 200 and "loading model" not in response_text and "error" not in response_text:
                print(f"'{model_name}' server on port {port} is ready for inference! 🎉", file=sys.stderr)
                return True
            elif response.status_code == 400 and "loading model" in response_text:
                print(f"'{model_name}' is still loading (400 response with 'loading model')...", file=sys.stderr)
            else:
                print(f"'{model_name}' server responded with status {response.status_code}. Response (first 100 chars): {response.text.strip()[:100]}", file=sys.stderr)
            
        except requests.exceptions.ConnectionError:
            print(f"Connection refused during dummy request for {model_name}. Server might be resetting or still starting...", file=sys.stderr)
        except requests.exceptions.Timeout:
            print(f"Dummy request for {model_name} timed out ({time.time() - start_time:.1f}/{timeout:.1f}s). Still loading?", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error during dummy request for {model_name}: {e}", file=sys.stderr)
        
        time.sleep(5) # Wait a bit longer between inference readiness checks, as model loading can be slow
    
    print(f"Error: '{model_name}' server on port {port} did not become ready for inference within {timeout} seconds. Final response was: {response_text if 'response_text' in locals() else 'No response'}", file=sys.stderr)
    return False

def activate_model_on_demand(model_name):
    """
    Activates a model by launching its llama-server process if it's not already running.
    If a previous process for this model is still active, it will be terminated first.
    Updates the registered_models dictionary with port and process info.
    Returns True on success, False on failure.
    """
    model_info = registered_models.get(model_name)
    if not model_info:
        print(f"Error: Model '{model_name}' is not registered.", file=sys.stderr)
        return False

    # Check if this model is already running and if so, terminate it for a fresh start
    if 'process' in model_info:
        if model_info['process'].poll() is None: # Process is still running
            print(f"Info: Terminating existing '{model_name}' server (PID {model_info['process'].pid}) for a fresh launch.", file=sys.stderr)
            model_info['process'].terminate()
            try:
                model_info['process'].wait(timeout=10) # Give it some time to terminate
            except subprocess.TimeoutExpired:
                print(f"Warning: '{model_name}' server (PID {model_info['process'].pid}) did not terminate gracefully. Killing.", file=sys.stderr)
                model_info['process'].kill()
        
        # Clean up process and port info regardless, to ensure a clean state before re-launch
        if 'process' in model_info: del model_info['process']
        if 'port' in model_info: del model_info['port']

    gguf_path = model_info.get('gguf_path')
    if not gguf_path or not os.path.exists(gguf_path):
        print(f"Error: GGUF file path not found or invalid for model '{model_name}': {gguf_path}", file=sys.stderr)
        return False

    port = find_free_port()
    process = run_llama_server_in_background(gguf_path, model_name, port)
    if not process:
        return False # Failed to launch process

    # Update registered_models with the new process and port
    registered_models[model_name]['port'] = port
    registered_models[model_name]['process'] = process

    # Wait for the server to be ready before proceeding
    if not wait_for_server_ready(port, model_name):
        print(f"Failed to activate model '{model_name}'. Terminating unresponsive process.", file=sys.stderr)
        process.terminate() # Terminate the unresponsive process
        # Clean up process and port info to allow re-attempt later
        if 'process' in registered_models[model_name]: del registered_models[model_name]['process']
        if 'port' in registered_models[model_name]: del registered_models[model_name]['port']
        return False
    
    return True


def is_port_in_use(port, host='127.0.0.1'):
    """Checks if a given TCP port is currently in use on a specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow reuse of local addresses
            s.bind((host, port)) # Try to bind to the port
            return False # If successful, port is free
        except OSError:
            return True # If binding fails, port is in use

class LlamaRouter(http.server.BaseHTTPRequestHandler):
    """
    A reverse proxy that routes incoming HTTP requests to the correct
    llama-server instance, supporting both OpenAI-like API calls
    (model in JSON body or specific /v1/models paths) and custom routing
    (model name as first path segment).
    Models are loaded on demand.
    """

    def _forward_request(self, method, model_name, forwarded_path, body=None):
        """
        Helper method to forward an HTTP request to a target URL with
        exponential backoff for retries.
        model_name: The identified name of the model (e.g., "deepseek-r1.gguf")
        forwarded_path: The path to send to the backend llama-server (e.g., "/v1/chat/completions")
        """
        # Ensure the model is active before attempting to forward
        if not activate_model_on_demand(model_name):
            self.send_error(503, f"Failed to activate model '{model_name}'. Service Unavailable.")
            return

        target_port = registered_models[model_name]['port']
        # Construct the target URL using the backend's port and the rewritten path
        # Backend llama-server instances are still bound to localhost for security/isolation
        target_url = f"http://localhost:{target_port}{forwarded_path}"
        
        # Preserve original query parameters
        parsed_original_path = urlparse(self.path)
        if parsed_original_path.query:
            target_url += f"?{parsed_original_path.query}"

        print(f"Routing {method} request for model '{model_name}' to backend: {target_url}", file=sys.stderr)

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
                # INCREASED TIMEOUT FOR THE ACTUAL FORWARDED REQUEST
                response = requests.request(
                    method,
                    target_url,
                    headers=headers,
                    data=body,
                    stream=True, # Use stream=True to handle large responses efficiently
                    timeout=3600 # Increased to 1 hour (3600 seconds) for actual inference
                )
                break # Request successful, exit retry loop
            except requests.exceptions.ConnectionError as e:
                # Handle connection errors (e.g., backend server died after activation)
                print(f"Connection error to {target_url}: {e}. Retrying in {backoff_factor * (2 ** i):.1f} seconds...", file=sys.stderr)
                time.sleep(backoff_factor * (2 ** i)) # Exponential backoff
            except requests.exceptions.Timeout as e:
                # Handle timeouts
                print(f"Timeout connecting to {target_url}: {e}. Retrying...", file=sys.stderr)
                time.sleep(backoff_factor * (2 ** i)) # Exponential backoff
        else:
            # All retries failed
            self.send_error(504, f"Failed to connect to model server after {retries} retries or request timed out.")
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
        # Example: GET http://<router_ip>:11337/v1/models
        # Example: GET http://<router_ip>:11337/v1/models/model_id
        if path_segments and path_segments[0] == "v1" and len(path_segments) >= 2 and path_segments[1] == "models":
            # If it's just /v1/models or /v1/models/ (list all *registered* models,
            # indicating active state based on process.poll() status)
            if len(path_segments) == 2 or (len(path_segments) == 3 and not path_segments[2]):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                models_data = []
                print(f"DEBUG (GET /v1/models): Currently registered models: {list(registered_models.keys())}", file=sys.stderr)
                for name, info in registered_models.items():
                    # Determine if the model is currently active (process exists and is running)
                    is_active = 'process' in info and info['process'].poll() is None
                    models_data.append({
                        "id": name,
                        "object": "model",
                        "created": int(time.time()), # Mock creation time
                        "owned_by": "local",
                        "active": is_active, # Indicate if the model is currently loaded/active
                        "permission": [
                            {"id": f"model-perm-{name}", "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": False, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}
                        ],
                        "root": name,
                        "parent": None
                    })
                print(f"DEBUG (GET /v1/models): Listing {len(models_data)} registered models (active status shown).", file=sys.stderr)
                
                response_payload = {
                    "object": "list",
                    "data": models_data
                }
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))
                return
            # If it's /v1/models/<model_id> (get details for a specific model)
            elif len(path_segments) >= 3:
                model_id_from_path = path_segments[2]
                if model_id_from_path in registered_models:
                    model_info = registered_models[model_id_from_path]
                    is_active = 'process' in model_info and model_info['process'].poll() is None
                    response_model_info = {
                        "id": model_id_from_path,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                        "active": is_active, # Indicate if the model is currently loaded/active
                        "permission": [
                            {"id": f"model-perm-{model_id_from_path}", "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": False, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}
                        ],
                        "root": model_id_from_path,
                        "parent": None
                    }
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response_model_info).encode('utf-8'))
                    return
                else:
                    self.send_error(404, f"Model '{model_id_from_path}' not found.")
                    return
        
        # Case 2: Custom Routing - model name as first path segment
        # Example: GET http://<router_ip>:11337/deepseek-r1.gguf/v1/chat/completions
        if path_segments and path_segments[0] in registered_models:
            model_name_from_path = path_segments[0]
            # _forward_request will call activate_model_on_demand internally
            
            # Rewrite path: remove the model name segment for the backend server
            forwarded_path = '/' + '/'.join(path_segments[1:])
            if not forwarded_path.startswith('/'): # Ensure leading slash
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
        # Example: POST http://<router_ip>:11337/v1/chat/completions (model in body)
        if path_segments and path_segments[0] == "v1" and len(path_segments) >= 2 and \
           path_segments[1] in ["chat", "completions"]: # Covers /v1/chat/completions and /v1/completions
            
            model_name_from_body = request_payload.get("model")

            if not model_name_from_body:
                self.send_error(400, "For /v1/chat/completions or /v1/completions, the 'model' field is required in the JSON body.")
                return

            if model_name_from_body not in registered_models:
                self.send_error(404, f"Model '{model_name_from_body}' not registered. "
                                     f"Registered models: {', '.join(registered_models.keys())}")
                return

            # _forward_request will call activate_model_on_demand internally
            # Forward the exact OpenAI-standard path to the backend
            forwarded_path = parsed_path.path # e.g., /v1/chat/completions
            self._forward_request(method='POST', model_name=model_name_from_body, forwarded_path=forwarded_path, body=body)
            return
        
        # Case 2: Custom Routing - model name as first path segment for POST requests
        # Example: POST http://<router_ip>:11337/deepseek-r1.gguf/v1/chat/completions
        if path_segments and path_segments[0] in registered_models:
            model_name_from_path = path_segments[0]
            # _forward_request will call activate_model_on_demand internally
            
            # Rewrite path: remove the model name segment for the backend server
            forwarded_path = '/' + '/'.join(path_segments[1:])
            if not forwarded_path.startswith('/'): # Ensure leading slash
                forwarded_path = '/' + forwarded_path
            
            self._forward_request(method='POST', model_name=model_name_from_path, forwarded_path=forwarded_path, body=body)
            return

        # If neither standard OpenAI nor custom path routing matches
        self.send_error(404, "Unsupported POST endpoint or model not specified/found.")


def run_router():
    """Starts the auto-router on the specified fixed port (11337)."""
    router_port = 11337
    # Explicitly try to bind to '0.0.0.0' to check if the port is in use globally
    if is_port_in_use(router_port, host='0.0.0.0'): 
        print(f"Router is already running on port {router_port}. Skipping starting a new one.", file=sys.stderr)
        return

    print(f"Starting auto-router on port {router_port}...", file=sys.stderr)
    # Bind the server to '0.0.0.0' to listen on all available network interfaces
    server_address = ('0.0.0.0', router_port) 
    # Use ThreadingHTTPServer for better concurrency when handling multiple client requests
    httpd = http.server.ThreadingHTTPServer(server_address, LlamaRouter)
    router_thread = threading.Thread(target=httpd.serve_forever)
    router_thread.daemon = True # Allows the main thread to exit, which will also terminate this thread
    router_thread.start()
    print(f"Auto-router started on http://0.0.0.0:{router_port}. This means it's accessible from any IP address on your network. Check your firewall if issues occur. Press Ctrl+C to stop all services.", file=sys.stderr)

# --- Original functions (only minor adjustments for print statements) ---

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
    
    print(f"\nExecuting: {' '.join(shlex.quote(arg) for arg in command)}", file=sys.stderr)

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
    print(f"Fetching manifest from: {manifest_url}", file=sys.stderr)

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
                   m.get('mediaType') == "application/vnd.oci.image.manifest.v1+json"):
                    found_manifest_digest = m.get('digest')
                    break
            
            if found_manifest_digest:
                sub_manifest_url = f"https://registry.ollama.ai/v2/library/{model_name_base}/manifests/{found_manifest_digest}"
                print(f"Fetching specific manifest for model from: {sub_manifest_url}", file=sys.stderr)
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
        print(f"Downloading GGUF from: {download_url}", file=sys.stderr)
        print(f"Saving to: {output_filename}", file=sys.stderr)

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
        print("\nDownload complete! 🎉", file=sys.stderr)
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
    The 'serve' command now exclusively uses the auto-router with on-demand model loading.
    """
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  python sllama.py modelfile <filename>      - Run llama-cli using instructions from a Modelfile", file=sys.stderr)
        print("  python sllama.py run <gguf_file>          - Run a local GGUF model file", file=sys.stderr)
        print("  python sllama.py run-hug <huggingface_repo> - Run a model directly from Hugging Face", file=sys.stderr)
        print("  python sllama.py serve <model_name>=<gguf_file> [<model_name>=<gguf_file>...]", file=sys.stderr)
        print("     - Starts the auto-router on port 11337 and registers models for on-demand loading.", file=sys.stderr)
        print("  python sllama.py dl-from-ollama <model_id> - Download a GGUF model from Ollama's registry (e.g., 'llama3.2:latest' or 'qwen3')", file=sys.stderr)
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
        
        # Register models, but don't start them yet
        for model_arg in sys.argv[2:]:
            if '=' not in model_arg:
                print(f"Invalid argument format: '{model_arg}'. Expected <model_name>=<gguf_file>", file=sys.stderr)
                continue
            model_name, gguf_file = model_arg.split('=', 1)
            
            if not os.path.exists(gguf_file):
                print(f"Error: Model file '{gguf_file}' not found. Skipping registration.", file=sys.stderr)
                continue
            
            registered_models[model_name] = {'gguf_path': gguf_file}
            print(f"Model '{model_name}' registered for on-demand loading from '{gguf_file}'.", file=sys.stderr)
        
        if not registered_models:
            print("No models registered. Router will not serve any models.", file=sys.stderr)
            sys.exit(1)

        # Start the router
        run_router()

        # Keep the main thread alive so the background router thread and on-demand model processes can run
        try:
            while True:
                # Periodically clean up processes that have died
                for name, info in list(registered_models.items()): # Use list() to allow modification during iteration
                    if 'process' in info and info['process'].poll() is not None:
                        print(f"Info: Model '{name}' server (PID {info['process'].pid}) has terminated.", file=sys.stderr)
                        # Remove process and port info, keep gguf_path for potential re-activation
                        del registered_models[name]['process']
                        if 'port' in registered_models[name]: del registered_models[name]['port']
                time.sleep(5) # Check every 5 seconds
        except KeyboardInterrupt:
            print("\nShutting down all servers...", file=sys.stderr)
            # Terminate all running processes
            for info in registered_models.values():
                if 'process' in info and info['process'].poll() is None: # Check if still running before trying to terminate
                    info['process'].terminate()
            print("All services stopped. Goodbye! 👋", file=sys.stderr)
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
