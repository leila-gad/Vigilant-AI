# pipelines/webrtc.py

def get_webrtc_pipeline():
    
    pipeline = """
    rtph264pay config-interval=1 pt=96 !
    webrtcbin name=sendrecv
    """
    
    return pipeline