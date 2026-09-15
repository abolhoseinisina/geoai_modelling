class YOLOConfig:
    def __init__(self, device: str):
        if device == 'mac':
            self.name = 'mac'
            self.device = 'mps'
            self.epochs = 1
            self.batch_size = 2
            self.model = "yolo26n-seg.pt"
            self.workers = 0
            self.patience = 1

        elif device == 'pc':
            self.name = 'pc'
            self.device = 'cuda'
            self.epochs = 80
            self.batch_size = 8
            self.model = "yolo26x-seg.pt"
            self.workers = 4
            self.patience = 20

        else:
            raise ValueError('Invalid "device" parameter.')