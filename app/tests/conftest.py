import os

# Suppress third-party library update checks and telemetry during test runs
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
