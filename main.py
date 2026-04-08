import yaml
import threading
import time
from backend.bridge.ai_bridge import AIBridge
from backend.signaling.signaling_server import SignalingServer

def load_config():
    with open("configs/settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    with open("configs/cameras.yaml", encoding="utf-8") as f:
        cameras = yaml.safe_load(f)["cameras"]
    return settings, cameras[0] if cameras else {}

def main():
    config, camera = load_config()
    signaling_cfg = config.get("signaling", {})
    signaling = SignalingServer(signaling_cfg.get("host", "0.0.0.0"), signaling_cfg.get("port", 8080))
    sig_thread = threading.Thread(target=signaling.start, daemon=True)
    sig_thread.start()

    bridge = AIBridge(config, camera)
    bridge.start()

    print("Vigilant-AI running. Open frontend/index.html (or serve with nginx). Latency <200ms target.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()

if __name__ == "__main__":
    main()