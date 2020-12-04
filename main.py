import flask
from flask import Flask, request, render_template


# 점자 -> 한글

from PIL import Image
import os, glob, numpy as np

import cv2

# 인풋 : 자를 점자 이미지 / 저장될 경로
# 기본정로 : full='../img/dot_sample/dot7.jpg', seg_path='../img/dotimg/test3'
# 리턴 : 한 글자씩 자른 점자 이미지들이 있는 폴더 경로
from main_pkg_b2h.dot_noising_segmentation import segmentation, seg2blist

# 인풋 : 한 글자씩 자른 점자 이미지들이 있는 폴더 경로
# 리턴 : 01 이중리스트
from main_pkg_b2h.dot_classification import model_test


# --------------------------------------
# 한글 -> 점자

# 텍스트 이미지 노이즈 제거 함수 / 인풋:img(처리할 이미지경로), dst(처리된 이미지경로) / 반환:dst(노이즈제거 이미지경로)
from main_pkg_h2b.opencv_noise import denoising

# OCR 함수 / 인풋:im(처리할 이미지경로) / 반환:text
from main_pkg_h2b.ocr_test import ocrtext

dst = './img/textimg/book_result.jpg'

# --------------------------------------

full = '../img/noiseTest/dot9.jpg'  # 만일을 위한 fullpath (임시 이미지)
seg_path='./img/dotimg/test3'      # 건들지 말기!

# 만약 seg_path 폴더에 파일이 있으면 모두 삭제
import os
def removeAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)

import hgtk
import hbcvt

from nlp_test import b2h, b2c

import io

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        # file = request.files['image']
        # if not file: return render_template('index.html', label="No Files")

        file = request.files['image'].read()

        radio = request.form['chk_info']

        if radio == 'one' :
            if not file: return render_template('index.html', label="No Files")
            # npimg = np.fromstring(file, np.uint8)
            # img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            dotlist = []
            # dot02 = ''

            for i in range(2):
                removeAllFile(seg_path)
                one_path = segmentation(file, seg_path, case=i)
                print(one_path)

                if one_path != None:
                    dot01 = model_test(one_path)
                    dot02 = seg2blist(one_path)

                    # print(dot01)

                    dotlist.append(dot01)
                    if dot02 == None:
                        dotlist.append('None')
                    else :
                        dotlist.append(dot02)
                    print('**************dotlist******************')
                    print(dotlist)
                else :
                    dotlist.append('None')
                    dotlist.append('None')

            print('================================')
            print('< 한글 번역 결과 >')

            resultlist = []
            # dotlist.append(seg2)      # 반환되는 리스트???
            dot_kor = ''
            for dot in dotlist:
                if dot == 'None' :
                    dot_kor = '이미지에서 패턴을 찾을 수 없습니다.'

                else:
                    try :
                        # 01->유니코드 점자 변환
                        e = b2c.b2ccvt(dot)
                        print(e)

                        # 01->한글 변환
                        f = b2h.b2hcvt(dot)
                        print(f)

                        dot_kor = e + ' / ' + f

                    except :
                        dot_kor = '해당 점자를 번역할 수 없습니다.'
                resultlist.append(dot_kor)
            print(resultlist)
            # dotlist.append(seg2)      # 반환되는 리스트???

            # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
            # label = resultlist

            # 숫자가 10일 경우 0으로 처리
            # if label == '10': label = '0'

            # 결과 리턴
            if len(set(resultlist)) == len(resultlist):     #중복이 없으면
                pass
            e = ['이미지에서 패턴을 찾을 수 없습니다.','해당 점자를 번역할 수 없습니다.']

            ### 추천 알고리즘
            resultdict = {'a':resultlist[0], 'b':resultlist[1],'c':resultlist[2],'d':resultlist[3]}
            tempkey = []
            tempvalue = []
            tempdict = {}
            for key,value in resultdict.items():
                if value not in e:
                    tempkey.append(key)
                    tempvalue.append(value)
                else:
                    tempdict[key] = value           #오류 값들은 따로 잠시 모아두기
            for key, value in tempdict.items():
                resultdict.pop(key)
            bestkey = ''
            bestvalue = ''
            print(tempkey,tempvalue[1:])
            for a in range(len(tempkey)-1):
                if tempvalue[a] in tempvalue[a+1:]:
                    print("중복")
                    # resultdict[tempkey[a]] = "[추천!] "+tempvalue[a]
                    bestkey = tempkey[a]
                    bestvalue = tempvalue[a]

            # 나온 결과가 하나면 그걸 바로 추천 결과로
            if len(resultdict) == 1:
                for key, value in resultdict.items():
                    bestkey = key
                    bestvalue = value
            if bestkey == '':
                try:
                    if len(resultdict['b']) > len(resultdict['d']):
                        bestkey = 'b'
                        bestvalue = resultdict['b']
                    elif len(resultdict['b'])<len(resultdict['d']):
                        bestkey = 'd'
                        bestvalue = resultdict['d']
                except:
                    try:
                        resultdict.get('b')
                        bestkey = 'b'
                        bestvalue = resultdict['b']
                    except:
                        try:
                            bestkey = 'd'
                            bestvalue = resultdict['d']
                        except:
                            bestkey = '?'
                            bestvalue = '추천 드릴 값이 없습니다. 사진을 다시 찍어주세요.'
            for key, value in tempdict.items():
                resultdict[key] = value

            # if resultlist[0] not in e and resultlist[0] in resultlist[1:3]:
            #     resultlist[0] = '>>>>>'+resultlist[0]
            # elif resultlist[1] not in e and resultlist[1] in resultlist[2:3]:
            #     resultlist[1] = '>>>>>'+resultlist[1]
            # elif resultlist[2] not in e and resultlist[2] in resultlist[3]:
            #     resultlist[2] = '>>>>>'+resultlist[2]
            # else:
            #     if

            return render_template('index.html', label=resultdict['a'], label2=resultdict['b'], label3=resultdict['c'], label4=resultdict['d'], label0=bestkey, label00=bestvalue)

        else :
            npimg = np.fromstring(file, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

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
            text = text.replace('\n', '')
            text = text.replace('|', '')
            text = text.replace('=', '')
            text = text.replace('@', '')
            text = text.replace('\'', '')
            text = text.replace('~', '')
            text = text.replace('%', '')
            text = text.replace('}', '')
            text = text.replace('{', '')

            print(text)

            print("===========================")

            print('< 점자 변환 >')
            # 한글->01 변환
            dot01 = hbcvt.h2b.h2b(text)
            print(dot01)

            # 01->유니코드 점자 변환
            dotuni = b2c.b2ccvt(dot01)
            # dotuni = dotuni.replace(' ', '\n')
            print(dotuni)

            print('< 한글 역변환 >')

            # 역변환
            # 01->한글 변환
            text_inv = b2h.b2hcvt(dot01)
            print(text_inv)



            # 결과 리턴


            return render_template('index.html', label=text, label2=dotuni, label3='aaa', label4='aaa')


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # model = joblib.load('./model/model.pkl')
    # model = load_model('./model/multi_img_classification3.model')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)