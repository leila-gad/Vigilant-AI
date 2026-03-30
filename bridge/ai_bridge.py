from ai_backend.processor import process

class AIBridge:

    def __init__(self):
        """
        Bridge entre GStreamer et backend AI
        """
        pass


    def process_frame(self, frame):
        """
        reçoit une frame depuis GStreamer
        l'envoie au backend AI
        retourne la frame traitée
        """

        frame = process(frame)

        return frame