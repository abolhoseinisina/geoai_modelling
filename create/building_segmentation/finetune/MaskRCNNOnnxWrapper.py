import torch

class MaskRCNNOnnxWrapper(torch.nn.Module):
    def __init__(self, detection_model: torch.nn.Module):
        super().__init__()
        self.detection_model = detection_model

    def forward(self, image: torch.Tensor):
        detection = self.detection_model([image])[0]
        return detection["boxes"], detection["labels"], detection["scores"], detection["masks"]