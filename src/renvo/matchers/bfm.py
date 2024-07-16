from .matcher import Matcher
import numpy as np
import cv2

class BFMatcher(Matcher):
    
    def __init__(self) -> None:
        super().__init__()
        self.__matcher = cv2.BFMatcher_create(crossCheck=True)
        
    def match(self, a_descriptors: np.array, b_descriptors: np.array) -> tuple[cv2.DMatch]:
        """_summary_

        Args:
            a_descriptors (np.array): _description_
            b_descriptors (np.array): _description_

        Returns:
            tuple[cv2.DMatch]: _description_
        """
        return self.__matcher.match(a_descriptors, b_descriptors)
        
    
    
    