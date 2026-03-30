const ws = new WebSocket("ws://localhost:8765");
const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }]
});

pc.ontrack = (event) => {
    document.getElementById("video").srcObject = event.streams[0];
};

const dataChannel = pc.createDataChannel("ai-data");

dataChannel.onmessage = (event) => {
    const data = JSON.parse(event.data);
    drawBox(data.coords);
};

function drawBox(coords) {
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeRect(coords[0], coords[1],
                   coords[2]-coords[0],
                   coords[3]-coords[1]);
}

ws.onmessage = async (event) => {
    const data = JSON.parse(event.data);

    if (data.answer) {
        await pc.setRemoteDescription(data);
    }
};

async function start() {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    ws.send(JSON.stringify(offer));
}

start();