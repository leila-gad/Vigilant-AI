import asyncio
import websockets
import json

class SignalingServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.connections = {}

    async def handler(self, ws):
        sid = id(ws)
        self.connections[sid] = ws
        try:
            async for msg in ws:
                data = json.loads(msg)
                # Forward offer/answer/ICE to webrtc_pipeline (wired in main.py)
                print(f"Signaling: {data['type']}")
        finally:
            self.connections.pop(sid, None)

    def start(self):
        async def serve():
            async with websockets.serve(self.handler, self.host, self.port):
                await asyncio.Future()
        asyncio.run(serve())