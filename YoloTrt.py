from trt_infer.infer import TensorRTInfer
from ObjectDetector import ObjectDetector



class YoloTRT(ObjectDetector):
    def __init__(self, model_path, classes) -> None:
        super().__init__(model_path, classes)
        self.model = TensorRTInfer(self.model_path)
    
    def predict(self, batch):
        return self.model.infer(batch) 