import subprocess
import os
import signal
from typing import Literal

PORT = 8080

def serve_model( model_path: str, model_type: Literal["vllm", "gguf"], dtype: str = "float16", max_model_len: int = 4096):

    if model_type == "vllm":
        cmd = [ "vllm", "serve", model_path, "--port", str(PORT)]

    elif model_type == "gguf":
        cmd = ["llama.cpp/build/bin/llama-server", "--model",model_path,"--port", str(PORT),"--ctx-size", "2048"]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print(f"🚀 Starting {model_type} server:")
    print(" ".join(cmd))

    subprocess.Popen(cmd)
