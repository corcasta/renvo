from .matcher import Matcher
import cv2

class BFMatcher(Matcher):
    
    def __init__(self) -> None:
        super().__init__()
        self.__matcher = cv2.BFMatcher_create(crossCheck=True)
        
    def match(self, a_descriptors, b_descriptors):
        return self.__matcher.match(a_descriptors, b_descriptors)
        
    
    
    