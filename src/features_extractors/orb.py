from .feature_extractor import FeatureExtractor
import cv2
import numpy as np

class ORB(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.__extractor = cv2.ORB_create()
        
    def keypoints(self, img: np.array) -> tuple[cv2.KeyPoint]:
        """
        Detects keypoints in an image (first variant) or image set (second variant).

        Args:
            img (np.array): 
                image

        Returns:
            tuple[cv2.KeyPoint]:
                The detected keypoints
        """
        return self.__extractor.detect(img)
    
    
    def descriptors(self, img: np.array, kps: tuple[cv2.KeyPoint]) -> tuple[cv2.KeyPoint, np.array]: 
        """
        Computes the descriptors for a set of keypoints detected in an image (first variant) or image 
        set (second variant).

        Args:
            img (np.array): 
                Image
            kps (tuple[cv2.KeyPoint]): 
                Input tuple of keypoints. Keypoints for which a descriptor cannot be computed are removed. 
                Sometimes new keypoints can be added.

        Returns:
            tuple[cv2.KeyPoint, np.array]: Updated keypoints and computed descriptors.
        """
        return self.__extractor.compute(img, kps)