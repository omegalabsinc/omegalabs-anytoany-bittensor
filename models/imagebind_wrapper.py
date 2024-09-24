import os
import torch
from imagebind import imagebind_model

V2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".checkpoints", "videobind.pth")

class ImageBind:
    def __init__(self, v2=True):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.v2 = v2
        if self.v2:
            if not os.path.exists(V2_PATH):
                os.makedirs(os.path.dirname(V2_PATH), exist_ok=True)
                torch.hub.download_url_to_file(
                    "https://huggingface.co/jondurbin/videobind-v0.1/resolve/main/videobind.pth",
                    V2_PATH,
                    progress=True,
                )
            self.imagebind = torch.load(V2_PATH)
        else:
            self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()
        self.imagebind.to(self.device)
        