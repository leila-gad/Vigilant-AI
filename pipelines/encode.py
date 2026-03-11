# encode.py

def get_encode_pipeline():
    
    pipeline = """
    videoconvert !
    x264enc bitrate=2048 speed-preset=ultrafast tune=zerolatency !
    video/x-h264,profile=baseline
    """
    
    return pipeline