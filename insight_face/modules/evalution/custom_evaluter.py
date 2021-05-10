import cv2
import numpy as np 

class CustomEvaluter:
    def __init__(self, config: dict):

        self.qi = config["qi"]
        self.b = config["b"]
        self.di = config["di"]

        self.blur_threshold = config["blur_threshold"]
        self.ratio0_min = config["ratio0_min"]
        self.ratio0_max = config["ratio0_max"]
        self.ratio1_max = config["ratio1_max"]
        self.ratio2_min = config["ratio2_min"]
        self.ratio2_max = config["ratio2_max"]
    
    def get_blur_var(self, area: float) -> float:
        return self.qi/((1.0+self.b*self.di*area)**(1.0/max(self.b, 1.e-50)))

    def check_not_blur(self, image: np.ndarray) -> bool:
        if image is None or image.size == 0:
            return False

        real_notblur = cv2.Laplacian(image, cv2.CV_64F).var()
        standard_notblur = self.get_blur_var(image.shape[0]*image.shape[1])

        print("real_notblur", real_notblur)

        threshnotblur = real_notblur/standard_notblur
        print(threshnotblur)

        if threshnotblur < self.blur_threshold:
            return False
        else:
            return True
        return False

    def check_straight_face(self, image: np.ndarray, lm: list) -> bool:
        cnt = lm.reshape(5,2, order='F')
        left_eye = cnt[0]
        right_eye = cnt[1]
        nose = cnt[2]
        left_mouth = cnt[3]
        right_mouth = cnt[4]
        middle_eye = (left_eye+right_eye)/2
        middle_mouth = (left_mouth+right_mouth)/2
        
        disme2mm = np.linalg.norm(middle_eye - middle_mouth) #Standard

        #Yaw picth rotate
        disle2re = np.linalg.norm(left_eye - right_eye) 
        disn2line_memo = np.linalg.norm(np.cross(middle_mouth-middle_eye, middle_eye-nose))/np.linalg.norm(middle_mouth-middle_eye)
        disn2me = np.linalg.norm(nose- middle_eye)

        ratio0 = disle2re/disme2mm
        ratio1 = disn2line_memo/disme2mm
        ratio2 = disn2me/disme2mm

        print("Ratio:", ratio0, ratio1, ratio2)

        if ratio0 > self.ratio0_min and ratio0 < self.ratio0_max and ratio1 < self.ratio1_max and ratio2 > self.ratio2_min and ratio2 < self.ratio2_max:
            return True
        else:
            return False

        return False