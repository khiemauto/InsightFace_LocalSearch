from typing import Tuple
import cv2
import numpy as np

class CustomEvaluter:
    def __init__(self, config: dict):

        self.qi = config["qi"]
        self.b = config["b"]
        self.di = config["di"]

        self.illumination_threshold = config["illumination_threshold"]
        self.dark_threshold = config["dark_threshold"]
        self.light_threshold = config["light_threshold"]
        self.blur_threshold = config["blur_threshold"]
        self.ratio0_min = config["ratio0_min"]
        self.ratio0_max = config["ratio0_max"]
        self.ratio1_max = config["ratio1_max"]
        self.ratio2_min = config["ratio2_min"]
        self.ratio2_max = config["ratio2_max"]
    
    def get_blur_var(self, area: float) -> float:
        return self.qi/((1.0+self.b*self.di*area)**(1.0/max(self.b, 1.e-50)))

    # def check_not_blur(self, image: np.ndarray, faceSize:float) -> Tuple[bool, float]:
    #     if image is None or image.size == 0:
    #         return False, 0.0, 0.0, 0.0

    #     real_notblur = cv2.Laplacian(image, cv2.CV_64F).var()
    #     standard_notblur = self.get_blur_var(image.shape[0]*image.shape[1])

    #     # standard_notblur = 0.00280530*faceSize +  68.7142432
    #     print(real_notblur, standard_notblur)

    #     threshnotblur = real_notblur/standard_notblur
    #     # print(threshnotblur)

    #     if threshnotblur < self.blur_threshold:
    #         return False, threshnotblur, real_notblur, standard_notblur
    #     else:
    #         return True, threshnotblur, real_notblur, standard_notblur

    #     return False, threshnotblur, real_notblur, standard_notblur

    def check_illumination(self, image):
        if image is None or image.size == 0:
            return False, 0.0, False, False
        image = cv2.resize(image, (112,112))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # length of R available  range  of  gray  intensities  excluding  5%  of  the darkest  and  brightest  pixel
        sorted_gray = np.sort(gray.ravel())
        l = len(sorted_gray)
        cut_off_idx = l * 5 // 100
        r = sorted_gray[l-cut_off_idx] - sorted_gray[cut_off_idx]
        illuminate = np.round(r / 255, 2)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        v = hsv[:,:,2]
        mean = v.mean()
        print("mean", mean)

        isDark = False
        isLight = False
        if mean < self.dark_threshold:
            isDark = True
        elif mean > self.light_threshold:
            isLight = True

        isillumination = True        
        if illuminate < self.illumination_threshold:
            isillumination = False    

        return isillumination, illuminate, isDark, isLight

    def check_not_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        if image is None or image.size == 0:
            return False, 0.0

        image = cv2.resize(image, (112,112))

        real_notblur = cv2.Laplacian(image, cv2.CV_64F).var()
        standard_notblur = 300

        threshnotblur = real_notblur/standard_notblur
        # print("threshnotblur", threshnotblur)

        if threshnotblur < self.blur_threshold:
            return False, threshnotblur
        else:
            return True, threshnotblur
        return False, threshnotblur

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

    def detect_blur_fft(self, image, size=60, thresh=10):
        # grab the dimensions of the image and use the dimensions to
        # derive the center (x, y)-coordinates
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))

        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)

        # check to see if we are visualizing our output
        if True:
            # compute the magnitude spectrum of the transform
            magnitude = 20 * np.log(np.abs(fftShift))

            # display the original input image
            (fig, ax) = plt.subplots(1, 2, )
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Input")
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # display the magnitude image
            ax[1].imshow(magnitude, cmap="gray")
            ax[1].set_title("Magnitude Spectrum")
            ax[1].set_xticks([])
            ax[1].set_yticks([])

            # show our plots
            plt.show()

        # zero-out the center of the FFT shift (i.e., remove low
        # frequencies), apply the inverse shift such that the DC
        # component once again becomes the top-left, and then apply
        # the inverse FFT
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)

        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        return (mean <= thresh, mean)