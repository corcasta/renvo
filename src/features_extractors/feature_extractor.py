from abc import ABC, abstractmethod
import numpy as np

class FeatureExtractor(ABC):
    
    @abstractmethod
    def keypoints(self, img: np.array):
        pass
    
    @abstractmethod
    def descriptors(self, img: np.array, keypoints):
        pass