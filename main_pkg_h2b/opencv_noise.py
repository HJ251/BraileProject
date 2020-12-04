
# 글자 인식만을 위한 깔끔한 이미지를 추출

import cv2
import numpy as np

# 텍스트
img = cv2.imread('./img/textimg/picture05.PNG')
dst = './img/textimg/picture05_result.PNG'

def denoising(img, dst):
    # Gray Scale
    # cvtColor : 컬러 변환 함수
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enlarge 2x
    height, width = gray.shape
    gray_enlarge = cv2.resize(gray, (2*width, 2*height), interpolation=cv2.INTER_LINEAR)

    # Denoising
    # fastNlMeansDenoising 함수 : 그레이 이미지에 대하여 잡음 제거
    denoised = cv2.fastNlMeansDenoising(gray_enlarge, h=10, searchWindowSize=21, templateWindowSize=7)

    # Thresholding
    # 이미지 쓰레스홀딩 : 어느 한 이미지에서 이미지 픽셀값이 문턱값보다 크면 어떤 고정된 값으로 할당하고, 작으면 다른 고정된 값으로 할당함
    gray_pin = 196
    ret, thresh = cv2.threshold(denoised, gray_pin, 255, cv2.THRESH_BINARY)

    # 점자
    # 입체감 있는 곳에 많이 쓰임
    # thresh = cv2.Canny(denoised, 50,100)

    # inverting
    thresh[260:2090] = ~thresh[260:2090]

    # 그레이랑 쓰레싱한 결과 이미지 두 개를 붙여서 저장
    # result = np.hstack((gray_enlarge, thresh))
    # cv2.imwrite(dst, result)

    # 그레이 이미지를 저장 (추천)
    # cv2.imwrite(dst, gray_enlarge)

    # 디스노이즈한 이미지를 저장 (추천)
    # 텍스트 이미지에서 그림자 제거를 할 수 있다
    cv2.imwrite(dst, denoised)

    # 쓰레싱 이미지를 저장
    # 텍스트 이미지 처리에서 쓰레싱은 별로 좋은 방법이 아닌 것 같다.
    # cv2.imwrite(dst, thresh)

    return dst


