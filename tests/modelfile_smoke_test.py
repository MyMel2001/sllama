import os
import sys
import tempfile

# Ensure the project root (containing sllama.py) is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sllama import register_models_from_modelfile, registered_models

def run_smoke_test():
    with tempfile.TemporaryDirectory() as d:
        # Create dummy GGUF files
        gguf1 = os.path.join(d, "modela.gguf")
        gguf2 = os.path.join(d, "modelb.gguf")
        open(gguf1, "wb").close()
        open(gguf2, "wb").close()

        # Create a Modelfile that references the two GGUF files and includes PARAMETER and SYSTEM for the first model
        modelfile = os.path.join(d, "modelfile.modelfile")
        with open(modelfile, "w") as f:
            f.write("FROM ./modela.gguf\\n")
            f.write("PARAMETER temperature 0.5\\n")
            f.write("SYSTEM \"\"\"Some system prompt\"\"\"\\n")
            f.write("FROM ./modelb.gguf\\n")

        models = register_models_from_modelfile(modelfile)
        assert len(models) == 2, f"expected 2 models, got {len(models)}"
        assert models[0][0] == "modelfile.modelfile_0", f"unexpected first model name: {models[0][0]}"
        assert models[1][0] == "modelfile.modelfile_1", f"unexpected second model name: {models[1][0]}"
        assert models[0][1] == gguf1, f"expected first path {gguf1}, got {models[0][1]}"
        assert models[1][1] == gguf2, f"expected second path {gguf2}, got {models[1][1]}"

        # Check that extras were attached for the first modelfile-derived model
        extra_args = registered_models.get("modelfile.modelfile_0", {}).get("extra_args", [])
        expected_extras = ["--temperature", "0.5", "-sys", "Some system prompt"]
        assert extra_args == expected_extras, f"unexpected extra args for modelfile modelfile_0: {extra_args}"
        print("smoke test passed")

if __name__ == "__main__":
    run_smoke_test()
