import cv2
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from pandas import DataFrame as pd
from PIL import Image
import math
from numpy import random
import os

from main_pkg_b2h.dot_noising import *


def segmentation(full='../img/dot_sample/dot7.jpg', seg_path='./img/dotimg/test3', case = 0):
    print('im in')
    try :
        # a = full.rfind('/')

        # 경로에서 확장자 빼고 파일 이름만 추출
        # filename = full[a + 1:-4]
        filename = 'result'

        npimg = np.fromstring(full, np.uint8)

        # ------------------------------------------

        if case == 0 :
            # how1 - 연재님
            # img = cv2.imread(full)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

        elif case == 1:
            # # # how2 - 소정님 : 블러 세 번하고 adaptiveThreshold
            # img = cv2.imread(full,0)
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8)
            # plt.imshow(thresh)
            # plt.show()

        # -------------------------------------

        # how2-1. ADAPTIVE_THRESH_MEAN_C 평균값으로 해봄
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7)

        # # how3
        # # 기존 처리
        # img = cv2.imread(full)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)
        # ret, thresh = cv2.threshold(denoised, 190, 255, cv2.THRESH_BINARY_INV)
        # thresh[260:2090] = ~thresh[260:2090]

        # how4 - dot_noising_segmentation2.py (how3+변경 전 새그)

        # # how5 - 기존 처리에서 thresh 말고 Canny만 한거
        # img = cv2.imread(full)
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)
        # thresh = cv2.Canny(denoised, 50, 100)

        # # 함수로 만든 노이징 처리 - noising03
        #
        # dstpath = '../img/noiseTest/result/'
        # dst = dstpath + filename + '_result3.jpg'
        # thresh = noising03(full, dst)

        # ------------------------------------------
        x_dot = thresh.shape[0]
        # print(x_dot)
        y_dot = thresh.shape[1]
        # print(y_dot)

        if thresh[0][0] == 0:
            thresh = 255 - thresh

        # plt.imshow(thresh)

        # print(thresh)
        # plt.show()

        startX = x_dot  # 모든 ‘0’ 중 가장 작은 x 자리 ( 계속 비교 )
        startY = 0  # 첫 ‘0’이 나타난 y 자리 ( 처음 저장 1 번 )

        endX = 0  # 모든 ‘0’ 중 가장 큰 x 자리 ( 계속 비교 )
        endY = 0  # 마지막 ‘0’이 나타난 y 자리 ( 계속 저장하다가 마지막 값 )

        flag = 0  # 처음 시작점 파악을 위함
        dis_flag = 0  # 전체 열이 255인 경우
        dis_flag2 = 0

        cutpoint = []  # y열을 구분할 위치 저장
        dd = []
        for jdx in range(y_dot):
            for idx in range(x_dot):
                # print(thresh[idx][jdx])
                if thresh[idx][jdx] != 255:
                    if flag == 0:
                        startY = jdx  # startY : 첫 '0'이 나타난 y 위치(고정)
                        flag = 1  # 처음 시작점 이후를 구분
                    if endX < idx:
                        endX = idx
                    if startX > idx:
                        startX = idx
                    endY = jdx  # endY : 마지막 '0'을 찾기 위한 기본 값 설정(계속 저장 > 마지막 저장이 마지막 값)
                    dis_flag = 1
            if dis_flag != dis_flag2:
                if len(dd) == 2:
                    cutpoint.append(dd)
                    dd = []
                    dd.append(jdx)
                else:
                    dd.append(jdx)
            dis_flag2 = dis_flag
            dis_flag = 0
        cutpoint.append(dd)
        # print('[ %d, %d ]' % (startX, startY))
        # print('[ %d, %d ]' % (endX, endY))
        # print(cutpoint)

        # thresh = thresh[startX-2:endX+2, :]

        dotsize = 0
        dgapavg = 0
        temp = 0
        max = 0
        for dd in cutpoint:
            b = dd[-1]
            a = dd[0]
            dcha = b - a  # 점 사이즈
            dotsize += dcha

            if max < dcha:
                max = dcha
            if temp != 0:
                dgap = a - temp  # 점 간격
            temp = b
        avg = dotsize / len(cutpoint)
        dotsize = round(avg)
        print("1차 점 크기 평균: ", avg)
        print("1차 점 크기 최대: ", max)

        ######temp - 찌끄레기들 제거

        startX = x_dot  # 모든 ‘0’ 중 가장 작은 x 자리 ( 계속 비교 )
        startY = 0  # 첫 ‘0’이 나타난 y 자리 ( 처음 저장 1 번 )

        endX = 0  # 모든 ‘0’ 중 가장 큰 x 자리 ( 계속 비교 )
        endY = 0  # 마지막 ‘0’이 나타난 y 자리 ( 계속 저장하다가 마지막 값 )

        flag = 0  # 처음 시작점 파악을 위함
        dis_flag = 0  # 전체 열이 255인 경우
        dis_flag2 = 0

        cutpoint = []  # y열을 구분할 위치 저장
        dd = []
        for jdx in range(y_dot):
            for idx in range(x_dot):
                # print(thresh[idx][jdx])
                if thresh[idx][jdx] != 255:
                    if flag == 0:
                        startY = jdx  # startY : 첫 '0'이 나타난 y 위치(고정)
                        flag = 1  # 처음 시작점 이후를 구분
                    if endX < idx:
                        endX = idx
                    if startX > idx:
                        startX = idx
                    endY = jdx  # endY : 마지막 '0'을 찾기 위한 기본 값 설정(계속 저장 > 마지막 저장이 마지막 값)
                    dis_flag = 1
            if dis_flag != dis_flag2:
                if len(dd) == 2:
                    if dd[1] - dd[0] < avg / 2:
                        # print(dd[0], dd[1])
                        for d in range(dd[0], dd[1] + 1):
                            # print("d:", d)
                            thresh[:, d] = 255
                        dd = []
                        dd.append(jdx)
                    else:
                        cutpoint.append(dd)
                        dd = []
                        dd.append(jdx)
                else:
                    dd.append(jdx)
            dis_flag2 = dis_flag
            dis_flag = 0
        cutpoint.append(dd)
        # print('[ %d, %d ]' % (startX, startY))
        # print('[ %d, %d ]' % (endX, endY))
        # print(cutpoint)

        thresh = thresh[startX - 2:endX + 2, :]
        # plt.imshow(thresh)

        # print(thresh)
        # plt.show()
        #########temp/

        dotsize = 0
        dgapmin = 1000000
        temp = 0
        max = 0
        for dd in cutpoint:
            b = dd[-1]
            a = dd[0]
            dcha = b - a  # 점 사이즈
            dotsize += dcha

            if max < dcha:
                max = dcha
            if temp != 0:
                dgap = a - temp  # 점 간격
                if dgap > 0 and dgapmin > dgap:
                    dgapmin = dgap
            temp = b
        avg = dotsize / len(cutpoint)
        dotsize = int(round(avg))
        print("2차 점 크기 평균: ", avg)
        print("2차 점 크기 최대: ", max)
        print("2차 점 간격 최소: ", dgapmin)

        ###################이제 dotsize 임의 지정해서 다시 cutpoint 뽑기
        # print(cutpoint)
        dotsize = (((dgapmin + max) * 3) / 4)
        for dd in cutpoint:
            dd[0] = round(dd[-1] - dotsize)
        # print(cutpoint)

        dgapmin = 1000000
        dgaps = []
        temp = 0
        for dd in cutpoint:
            b = dd[-1]
            a = dd[0]
            dcha = b - a  # 점 사이즈
            if temp != 0:
                dgap = a - temp  # 점 간격
                dgaps.append(dgap)
                if dgap > 0 and dgapmin > dgap:
                    dgapmin = dgap
            temp = b
        onegapmin = 1000000
        # print(dgaps)
        # for i in range(len(dgaps)):
        #     if dgaps[i] == dgapmin:
        #         try:
        #             if onegapmin > dgaps[i + 1]:
        #                 onegapmin = dgaps[i + 1]
        #             if onegapmin > dgaps[i - 1]:
        #                 onegapmin = dgaps[i - 1]
        #         except:
        #             pass
        b = 1000000
        for i in range(len(dgaps) - 1):
            a = dgaps[i] + dgaps[i + 1]
            if b > a:
                b = a
                if dgaps[i] > dgaps[i + 1]:
                    onegapmin = dgaps[i]
                    dgapmin = dgaps[i + 1]
                else:
                    onegapmin = dgaps[i + 1]
                    dgapmin = dgaps[i]

        dotsize = int(dotsize)
        print("점 크기: ", dotsize)
        print("점 간격 최소: ", dgapmin)
        print("세트 간격 최소: ", onegapmin)

        one = [[]]
        whole = []
        for i in range(len(cutpoint)):
            one = []
            d1 = cutpoint[i]
            if i == len(cutpoint) - 2:
                # print("ㅇㅇㅇㅇ")
                d2 = cutpoint[i + 1]
                d3 = [d2[0] + dotsize + dgapmin + onegapmin, d2[1] + dotsize + dgapmin + onegapmin]
            elif i == len(cutpoint) - 1:
                d2 = [d1[0] + dotsize + dgapmin + onegapmin, d1[1] + dotsize + dgapmin + onegapmin]
                d3 = d2
            else:
                d2 = cutpoint[i + 1]
                d3 = cutpoint[i + 2]
            gap1 = d2[0] - d1[-1]
            gap2 = d3[0] - d2[-1]
            if len(whole) == 0 and gap1 >= onegapmin:  # 처음 시작하는데 한쪽열만 있으면
                if gap2 < onegapmin:  # 그 다음이 한 세트면 그냥 왼쪽에 점 찍어주면 되고
                    one = [[d1[0] - dgapmin - dotsize, d1[0] - dgapmin], d1]
                    a = 2.1
                elif gap1 > dgapmin * 2 + dotsize * 2 + onegapmin:  # 왼쪽점이면
                    one = [d1, [d1[1] + dgapmin, d1[1] + dgapmin + dotsize]]  # 오른쪽에 점 찍어주고..
                    a = 3.1
                # print(d1, d2, d3, a)
                whole.append(one)
                one = []
            if gap2 > gap1 and gap1 < onegapmin:  # d1 d2 간격이 좁으면 한 세트지
                one = [d1, d2]
                a = 1
            elif gap2 >= onegapmin:  # d1 d2 간격이 한 세트 간격이 아니고 d2 d3 간격이 세트 간의 간격 정도면
                if gap1 > (dotsize + onegapmin):  # 그리고 d1 d2 간격이 넓으면
                    one = [[d2[0] - dgapmin - dotsize, d2[0] - dgapmin], d2]  # 왼쪽에 점을 찍어주자
                    a = 2
                else:
                    one = [d2, [d2[1] + dgapmin, d2[1] + dgapmin + dotsize]]  # 아님 오른쪽에 점을 찍어주자
                    a = 3
                    cutpoint.pop(i + 1)  # 그리고 다음거도 이런거일수 있으니까 cutpoint에서 점을 걍 바꿔치기
                    cutpoint.insert(i + 1, one[-1])
            elif i == len(cutpoint) - 2:  # 끝쪽에서
                if gap1 < dotsize + onegapmin and gap1 >= onegapmin:  # d1 d2 간격이 세트 간 간격보단 크고 점 하나 드갈 사이즈는 아닌 경우
                    one = [d2, [d2[1] + dgapmin, d2[1] + dgapmin + dotsize]]  # 오른쪽에 점 찍어주기
                    a = 3.2
                elif gap1 >= dotsize + onegapmin + dgapmin:
                    one = [[d2[0] - dgapmin - dotsize, d2[0] - dgapmin], d2]  # 왼쪽에 점 찍어주기
                    a = 2.2

            if one != []:  # 한 세트가 채워지긴 했는데
                if len(whole) >= 1 and len(whole) < len(cutpoint):  # ?
                    if whole[-1][1][1] > one[0][0]:  # 문장에 넣어둔 마지막이 지금 한 세트랑 겹치면
                        if one[1][1] - one[0][0] < dotsize * 2 + dgapmin:  # 근데 지금 세트가 한 세트라고 하기엔 넘 작으면 얘가 잘못된거니까
                            continue
                        if a != 3 and a != 2:  # 그리고 이게 오른쪽 점 찍은 거 아니면
                            whole.pop(-1)  # 앞에 세트는 빼고 지금 세트를 넣자
                            a = -1
                        else:  # 근데 만약 앞 세트가 오른쪽 점 찍은 세트면
                            continue  # 걍 건너뛰자
                # print(d1, d2, d3, a)
                whole.append(one)
            if gap1 >= (dotsize * 2 + dgapmin + onegapmin * 2):  # 끝쪽 아니고 하나짜리도 아니면서 갭이 왕왕크면 공백 점자를 만들어주자
                one = [[d1[1] + onegapmin, d1[1] + onegapmin + dotsize],
                       [d1[1] + onegapmin + dotsize + dgapmin, d1[1] + onegapmin + dotsize * 2 + dgapmin]]
                a = 4
                print("공백이다 공백")
                whole.insert(-1, one)
        # print(whole)
        # cha = []
        # one = cutpoint[0][0]
        # for minus in range(1, len(cutpoint)):
        #     cha.append(cutpoint[minus]-one)
        #     one = cutpoint[minus]

        # cutpoint2 = np.asarray(cutpoint) - startY
        # print(cutpoint2)
        # print(cha)

        # thresh_new = thresh[startX-1:endX+1, startY-1:endY+1]
        #
        # plt.imshow(thresh)
        imsave = []
        for one in whole:
            try:
                plt.axvline(x=one[0][0], color='r', linestyle='--', linewidth=0.5)
                plt.axvline(x=one[1][1], color='b', linestyle='--', linewidth=0.5)
                plt.axvline(x=one[0][1], color='r', linestyle='--', linewidth=0.1)
                plt.axvline(x=one[1][0], color='b', linestyle='--', linewidth=0.1)
                imsave.append([one[0][0], one[1][1]])
            except:
                pass
        print("이건?")
        # plt.show()

        padding = np.full((thresh.shape[0], 2), 255)

        # 자르기
        for idx in range(0, len(imsave)):
            # print(imsave[idx][0])
            savename = seg_path + '/' + filename + '_' + str(idx).zfill(2) + '.jpg'

            image = thresh[:, int(imsave[idx][0]):int(imsave[idx][1])]
            image = np.hstack([image, padding])

            cv2.imwrite(savename, image)

        return seg_path

    except Exception as err :
        pass

# segmentaion end

def seg2blist(one_path):
    try :
        result = []
        if os.path.exists(one_path):
            for file in os.scandir(one_path):
                img = cv2.imread(file.path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
                # print(file)
                h3 = len(thresh)
                h1 = int(h3/3)
                h2 = h1*2
                v2 = len(thresh[0])
                v1 = int(v2/2)

                vflag, hflag = 0,0

                vstart1, vend1, vstart2 = 0,0,0

                hstart1, hend1, hstart2, hend2, hstart3 = 0,0,0,0,0

                # print(thresh)
                d1, d2, d3, d4, d5, d6 = 0,0,0,0,0,0
                for a in range(h3):
                    if hflag == 0 and sum(thresh[a,:]) != 0:
                        hstart1 = a
                        hflag = 1
                    elif hflag == 1 and sum(thresh[a,:]) == 0:
                        hend1 = a
                        hflag = 2
                    elif hflag == 2 and sum(thresh[a,:]) != 0:
                        hstart2 = a
                        hflag = 3
                    elif hflag == 3 and sum(thresh[a,:]) == 0:
                        hend2 = a
                        hflag = 4
                    elif hflag == 4 and sum(thresh[a,:]) != 0:
                        hstart3 = a
                        hflag = 5
                ###########h끝났음 한번 돌려보고~~
                if hflag == 0:  # 점이 없는가
                    # 공백                    #그럼 공백
                    pass
                elif hflag > 4:  # 점이 세개가 잡히는가
                    h1 = hstart2  # 그럼 쉽게 2번째, 3번째 점 시작점에서 가르기
                    h2 = hstart3
                elif hflag > 2:  # 점이 두개가 잡히는가
                    if (hend1 - h1) + (hstart1 - h1) < 0:  # 그리고 첫 점이 위쪽으로 치우쳐져 있는가
                        h1 = hend1 + 1  # 그렇다면 위쪽 점 끝에서 가르기
                        if (hend2 - h2) + (hstart2 - h2) < 0:  # 그리고 두번째 점도 위쪽인가
                            h2 = hend2 + 1
                        else:
                            h2 = hstart2-1
                    elif (hend1 - h2) + (hstart1 - h2) < 0:  # 그게 아니면 위쪽 점 끝에서 윗 라인
                        h2 = hend1 + 1
                    else:
                        h1 = hstart1-1  # 아님 오른쪽점 시작에서 가르기
                else:  # 점이 한개인가
                    if (hend1 - h1) + (hstart1 - h1) < 0:  # 점 위쪽?
                        h1 = hend1 + 1  # 그렇다면 위쪽 점 끝에서 가르기
                    elif (hend1 - h2) + (hstart1 - h2) < 0:  # 두번재 라인보다도 위쪽?
                        h2 = hend1 + 1
                    else:
                        h2 = hstart1-1
                # print(hflag)

                #####vflag 시작
                for b in range(v2):
                    #print(">>>>>>>>>>>>>>",sum(thresh[:,b]))
                    if vflag == 0 and sum(thresh[:,b]) != 0:
                        vstart1 = b
                        vflag = 1
                    elif vflag == 1 and sum(thresh[:,b]) == 0:
                        vend1 = b
                        vflag = 2
                    elif vflag == 2 and sum(thresh[:,b]) != 0:
                        vstart2 = b
                        vflag = 3
                # print(vflag)
                ##############vflag도 돌려보고~~~
                if vflag == 0:              #점이 없는가
                    #공백                    #그럼 공백
                    pass
                elif vflag == 3:                #점이 두개가 잡히는가
                    v1 = vend1+1                #그럼 쉽게 1번째 점 끝에서 가르기
                elif vflag == 2:
                    if (vend1-v1)+(vstart1-v1)<0:       #점이 왼쪽으로 치우쳐져 있는가
                        v1 = vend1+1            #그렇다면 왼쪽점 끝에서 가르기
                    else:
                        v1 = vstart1-1            #아님 오른쪽점 시작에서 가르기
                vflag = 0

                # print(v1, v2, h1, h2, h3)
                # print(thresh[h2:,v1:])
                if sum(sum(thresh[:h1,:v1])) > 0:
                    d1 = 1
                if sum(sum(thresh[h1:h2,:v1])) > 0:
                    d2 = 1
                if sum(sum(thresh[h2:,:v1])) > 0:
                    d3 = 1
                if sum(sum(thresh[:h1,v1:])) > 0:
                    d4 = 1
                if sum(sum(thresh[h1:h2,v1:])) > 0:
                    d5 = 1
                if sum(sum(thresh[h2:,v1:])) > 0:
                    d6 = 1

                result.append([d1,d2,d3,d4,d5,d6])
        return result
    except Exception as arr :
        return None