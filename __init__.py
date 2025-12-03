from .whisperx_node import WhisperXNode

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "WhisperXNode": WhisperXNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
