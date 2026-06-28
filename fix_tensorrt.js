module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: "Installing TensorRT 10.4 into the existing env...\nThis installs the meta package AND the -libs/-bindings subpackages that actually contain the runtime DLLs (nvinfer_10.dll etc.) onnxruntime needs for the TensorRT execution provider.\nVersion 10.4 is installed to run TensorRT acceleration.\nThis may take a few minutes — the libs package is ~1 GB."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --extra-index-url https://pypi.nvidia.com/ tensorrt-cu12==10.4.0 tensorrt-cu12-libs==10.4.0 tensorrt-cu12-bindings==10.4.0"
        ]
      }
    },
    {
      method: "log",
      params: {
        text: "Done! TensorRT is now installed.\nRestart the app (Stop → Start) and you should see:\n  Using provider [('TensorrtExecutionProvider', ...)] - Device:cuda"
      }
    }
  ]
}
