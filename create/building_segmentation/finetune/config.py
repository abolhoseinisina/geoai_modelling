class FineTuneConfig:
    def __init__(self, device: str):
        if device == 'mac':
            self.name = 'mac'
            self.pretrained_model_path = "../../models/building_footprints_usa.pth"
       
            self.device = 'cpu'
            self.epochs = 1
            self.batch_size = 2
            self.learning_rate = 0.002
            self.num_workers = 0
            self.pin_memory = False 
        
        elif device == 'pc':
            self.name = 'pc'
            self.pretrained_model_path = "../../models/building_footprints_usa.pth"
            self.device = 'cuda'
            self.epochs = 24
            self.batch_size = 4
            self.learning_rate = 0.005
            self.num_workers = 4
            self.pin_memory = True 
        
        else:
            raise ValueError('Invalid "device" parameter.')