from .whispercpp_node import WhisperCPPNode

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "WhisperCPPNode": WhisperCPPNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperCPPNode": "WhisperCPP Transcription",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
