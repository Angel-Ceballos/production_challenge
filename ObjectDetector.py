from abc import ABC, abstractmethod

# load weights and classes
class ObjectDetector(ABC):
    def __init__(self, model_path, classes) -> None:
        self.model_path = model_path
        self.classes = classes
        
    @abstractmethod
    def predict(self):
        pass

