module.exports = {
  run: [
    {
      method: "log",
      params: {
        text: "Installing TensorRT Python package into the existing env...\nThis provides the runtime DLLs (nvinfer_10.dll etc.) that onnxruntime needs for the TensorRT execution provider.\nThis may take a few minutes — the package is several hundred MB."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          "PYTHONHTTPSVERIFY": "0"
        },
        message: [
          "python -m pip install --extra-index-url https://pypi.nvidia.com/ tensorrt==10.4.0"
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
