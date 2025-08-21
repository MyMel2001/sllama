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

# Dictionary to store information about running llama-server processes
# Keys are model names, values are dictionaries with 'port' and 'process'
running_models = {}

def find_free_port():
    """Finds a free port on the system by binding to a random port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_command_in_background(executable, args, name, port):
    """
    Executes an external command in the background without waiting.
    Stores the process object for later management.
    """
    command = [executable] + args
    print(f"\nExecuting '{name}' in background: {' '.join(shlex.quote(arg) for arg in command)}")
    try:
        # Use Popen to run the command asynchronously
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class LlamaRouter(http.server.BaseHTTPRequestHandler):
    """
    A simple reverse proxy that routes requests to the correct llama-server instance.
    """
    def do_GET(self):
        # Handle the special /list request to list running models
        if self.path == "/list":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            model_list = [
                {'model': name, 'port': info['port'], 'status': 'running' if info['process'].poll() is None else 'stopped'}
                for name, info in running_models.items()
            ]
            self.wfile.write(json.dumps(model_list).encode('utf-8'))
            return

        # Route other GET requests
        self.handle_proxy_request()

    def do_POST(self):
        # Route POST requests
        self.handle_proxy_request()

    def handle_proxy_request(self):
        # Parse the requested path to get the model name
        parsed_path = urlparse(self.path)
        path_parts = parsed_path.path.strip('/').split('/')
        if not path_parts:
            self.send_error(404, "Invalid URL format. Expected /<model_name>/...")
            return

        model_name = path_parts[0]
        if model_name not in running_models:
            self.send_error(404, f"Model '{model_name}' not found or not running.")
            return

        target_port = running_models[model_name]['port']
        target_path = '/' + '/'.join(path_parts[1:])
        target_url = f"http://localhost:{target_port}{target_path}{parsed_path.query}"
        
        try:
            # Forward the request to the target server
            headers = dict(self.headers)
            headers['Host'] = f"localhost:{target_port}" # Correct Host header for the target
            
            # Read the request body for POST requests
            body = None
            if self.command == 'POST':
                content_length = int(headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)

            response = requests.request(
                self.command,
                target_url,
                headers=headers,
                data=body,
                stream=True,
                timeout=300 # Set a reasonable timeout
            )
            
            # Forward the response back to the client
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.content)

        except requests.exceptions.RequestException as e:
            self.send_error(502, f"Failed to forward request to model server: {e}")
        except Exception as e:
            self.send_error(500, f"An internal error occurred: {e}")

def run_router():
    """Starts the auto-router on the specified port."""
    router_port = 11337
    if is_port_in_use(router_port):
        print(f"Router is already running on port {router_port}. Skipping.")
        return

    print(f"Starting auto-router on port {router_port}...")
    server_address = ('', router_port)
    httpd = http.server.HTTPServer(server_address, LlamaRouter)
    router_thread = threading.Thread(target=httpd.serve_forever)
    router_thread.daemon = True # Allow the main thread to exit
    router_thread.start()
    print("Auto-router started. Press Ctrl+C to stop all services.")

def parse_modelfile(filename):
    """
    (Original function, no changes needed for this prompt)
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
    (Original function, unchanged)
    """
    command = [executable] + args
    print(f"\nExecuting: {' '.join(shlex.quote(arg) for arg in command)}")
    try:
        process = subprocess.Popen(command)
        process.wait()
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
    (Original function, unchanged)
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
        for layer in manifest.get('layers', []):
            if 'digest' in layer and 'mediaType' in layer:
                if layer['mediaType'] == "application/vnd.ollama.image.model" or \
                   layer['mediaType'].startswith("application/vnd.ollama.image.model.") or \
                   layer['mediaType'] == "application/octet-stream":
                    gguf_digest = layer['digest']
                    print(f"Found GGUF digest in layer with mediaType: {layer['mediaType']}", file=sys.stderr)
                    break
        if not gguf_digest:
            config_digest = manifest.get('config', {}).get('digest')
            if config_digest and config_digest.startswith("sha256:"):
                gguf_digest = config_digest
                print("Found GGUF digest in config.", file=sys.stderr)
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
                sys.stdout.write(f"\rDownloading: {readsofar / (1024*1204):.2f}MB downloaded...")
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
        print("  python sllama.py modelfile <filename>      - Run llama-cli using instructions from a Modelfile")
        print("  python sllama.py run <gguf_file>          - Run a local GGUF model file")
        print("  python sllama.py run-hug <huggingface_repo> - Run a model directly from Hugging Face")
        print("  python sllama.py dl-from-ollama <model_id> - Download a GGUF model from Ollama's registry (e.g., 'llama3.2:latest' or 'qwen3')")
        print("  python sllama.py serve <model_name>=<gguf_file> [<model_name>=<gguf_file>...]")
        print("     - Starts one or more llama-server instances on random ports, with an auto-router on port 11337.")
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
    elif command == "dl-from-ollama":
        if len(sys.argv) != 3:
            print("Usage: python sllama.py dl-from-ollama <model_id>", file=sys.stderr)
            sys.exit(1)
        model_id = sys.argv[2]
        download_from_ollama(model_id)
    elif command == "serve":
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
                ["-m", shlex.quote(gguf_file), "--port", str(port)],
                model_name,
                port
            )
        
        # Keep the main thread alive so the background servers and router can run
        try:
            while True:
                threading.Event().wait(3600)
        except KeyboardInterrupt:
            print("\nShutting down all servers...")
            # Terminate all running processes
            for info in running_models.values():
                info['process'].terminate()
            print("All services stopped. Goodbye! 👋")
            sys.exit(0)
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print("Supported commands are: modelfile, run, run-hug, serve, dl-from-ollama, serve-auto", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()