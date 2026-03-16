import cv2

class AIBridge:

    def __init__(self, ai_backend=None):
        """
        ai_backend : fonction ou module qui traite les frames
        """
        self.ai_backend = ai_backend


    def process_frame(self, frame):
        """
        reçoit une frame depuis GStreamer
        l'envoie au backend AI
        retourne la frame traitée
        """

        if self.ai_backend is not None:
            frame = self.ai_backend(frame)

        return frame