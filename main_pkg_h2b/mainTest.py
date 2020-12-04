

# < 한글 to 점자 프로세스 >
#
# 이미지 인풋
#       |
# 이미지의 노이즈를 제거
# (thresing 제외한 denoising 단계까지만)
# opencv_noise.py
#       |
# 노이즈 제거한 이미지 인풋
#       |
# 주어진 이미지를 한글로 OCR
# ocr_test.py
#       |
# 텍스트 인풋
#       |
# 점자로 자연어 처리 h2b
# chojungjong.py
#       |
# 점자 추출

# ==========================================


import cv2

# 텍스트 이미지 노이즈 제거 함수 / 인풋:img(처리할 이미지경로), dst(처리된 이미지경로) / 반환:dst(노이즈제거 이미지경로)
from main_pkg_h2b.opencv_noise import denoising

# OCR 함수 / 인풋:im(처리할 이미지경로) / 반환:text
from main_pkg_h2b.ocr_test import ocrtext

# 한글과 점자 조합
# from main_pkg_h2b.chojungjong import *

img = cv2.imread('../img/textimg/book (7).jpg')
dst = '../img/textimg/book (7)_result.jpg'

denois_img = denoising(img, dst)

print("===========================")

# 이미지에 있는 텍스트 출력
print('< 이미지에 있는 텍스트 추출 >')
text = ocrtext(denois_img)
print(text)

print("===========================")

# 점자와 매칭되지 않는 문자들을 제거해줌
# 불용어 처리?
text = text.strip()
text = text.replace('\n','')
text = text.replace('|','')
text = text.replace('=','')
text = text.replace('@','')
text = text.replace('\'','')
text = text.replace('~','')
text = text.replace('%','')

print(text)

print("===========================")

import hgtk
import hbcvt

from nlp_test import b2h, b2c


print('< 점자 변환 >')
# 한글->01 변환
dot01 = hbcvt.h2b.h2b(text)
print(dot01)

# 01->유니코드 점자 변환
dotuni = b2c.b2ccvt(dot01)
print(dotuni)


print('< 한글 역변환 >')

####### 에러 구간 #########
# 역변환
# 01->한글 변환
text_inv = b2h.b2hcvt(dot01)
print(text_inv)



