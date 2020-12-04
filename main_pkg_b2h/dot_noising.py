
import cv2
from matplotlib import pyplot as plt

# # 예시들
# # 노이징할 이미지 경로
# imgpath = '../img/dot_sample/dot18.JPG'
# # 노이징 결과 이미지 경로
# dst = '../img/dot_sample/dot18_result.JPG'

# 그레이 -> 쓰레싱(190,255)
def noising01(imgpath, dst):
    img = cv2.imread(imgpath)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(gray,190,255,cv2.THRESH_BINARY_INV)

    x_dot = thresh2.shape[0]
    # print(x_dot)
    y_dot = thresh2.shape[1]
    # print(y_dot)

    if thresh2[0][0] == 0 :
        thresh2 = 255 - thresh2

    show_imwrite_output(dst, thresh2)


# 그레이 -> 쓰레싱(자동으로 수치 조절)
def noising02(imgpath, dst):
    img = cv2.imread(imgpath)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    show_imwrite_output(dst, thresh1)


# 그레이 -> 디노이즈 -> 쓰레싱(165,255)
def noising03(imgpath, dst):
    img = cv2.imread(imgpath)

    # Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoising
    # fastNlMeansDenoising 함수 : 그레이 이미지에 대하여 잡음 제거
    denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)

    # Thresholding
    # 이미지 쓰레스홀딩 : 어느 한 이미지에서 이미지 픽셀값이 문턱값보다 크면 어떤 고정된 값으로 할당하고, 작으면 다른 고정된 값으로 할당함
    gray_pin = 165
    ret, thresh = cv2.threshold(denoised, gray_pin, 255, cv2.THRESH_BINARY_INV)

    # inverting
    thresh[260:2090] = ~thresh[260:2090]

    show_imwrite_output(dst, thresh)

    return thresh


# 그레이 -> 디노이즈
def noising04(imgpath, dst):
    img = cv2.imread(imgpath)

    # Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Enlarge 2x
    height, width = gray.shape

    # Denoising
    # fastNlMeansDenoising 함수 : 그레이 이미지에 대하여 잡음 제거
    denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)

    show_imwrite_output(dst, denoised)


# 그레이
def noising05(imgpath, dst):
    img = cv2.imread(imgpath)

    # Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Enlarge 2x
    height, width = gray.shape
    gray_enlarge = cv2.resize(gray, (2*width, 2*height), interpolation=cv2.INTER_LINEAR)

    show_imwrite_output(dst, gray_enlarge)

# 케니
def noising06(imgpath, dst):
    pass
    # img_canny = cv2.Canny(circles, 50, 100)

# 결과 이미지 출력하고 저장하는 함수
def show_imwrite_output(dst, output):
    plt.imshow(output)
    cv2.imwrite(dst, output)
    # print(output + ' 저장 완료!')























