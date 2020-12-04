import hgtk
import hbcvt

# 점자로 번역 (array 형태?? 중첩 리스트)
# c = hbcvt.h2b.text("그렇지않아")
# print(c)
# print("--------")
#
# #점자로 번역
# c = hbcvt.h2b.h2b(a)
# print(c)
# print("--------")
#
#
# #한글 문자로 치환
# d = hbcvt.h2b.b2h(aa)
# print(d)
#
# #치환된 문자를 조합
# e = hgtk.text.compose(d)
# print(e)


import numpy as np
from hgtk import text
from hgtk.text import compose

from nlp_test import b2h, b2c

#이름이 뭐지?
a = [[1,0,1,0,1,0], [0,0,0,0,1,0], [0,1,0,1,0,1], [0,1,0,0,0,1], [1,0,1,0,1,0],[0,0,0,0,0,0],
       [1,0,0,0,1,0], [1,1,1,1,0,0], [0,0,0,1,0,1], [1,0,1,0,1,0], [0,1,1,0,0,1]]

#아, 달이 밝구나!
b = [[1,1,0,0,0,1], [0,0,0,0,1,0], [0,0,0,0,0,0],
       [0,1,0,1,0,0], [0,1,0,0,0,0], [1,0,1,0,1,0],[0,0,0,0,0,0],
       [0,0,0,1,1,0],[0,1,0,0,0,0], [1,0,0,0,0,0], [0,0,0,1,0,0], [1,0,1,1,0,0],
       [1,0,0,1,0,0], [0,1,1,0,1,0]]

#이게 누구야!
c = [[1,0,1,0,1,0], [0,0,0,1,0,0], [1,0,1,1,1,0], [0,0,0,0,0,0],
       [1,0,0,1,0,0], [1,0,1,1,0,0],
       [0,0,0,1,0,0], [1,0,1,1,0,0], [0,0,1,1,1,0], [0,1,1,0,1,0]]

#근면, 검소, 협동은 우리 겨레의 미덕이다.
d = [[0,0,0,1,0,0,], [1,0,1,0,1,1], [1,0,0,0,1,0], [1,0,0,0,0,1], [0,0,0,0,1,0],[0,0,0,0,0,0],
    [0,0,0,1,0,0], [0,1,1,1,0,0], [0,1,0,0,0,1], [0,0,0,0,0,1], [1,0,1,0,0,1], [0,0,0,0,1,0],[0,0,0,0,0,0],
    [0,1,0,1,1,0], [1,0,0,0,1,1], [1,1,0,0,0,0], [0,1,0,1,0,0], [1,1,1,1,1,1], [1,0,1,0,1,1],[0,0,0,0,0,0],
    [1,0,1,1,0,0], [0,0,0,0,1,0], [1,0,1,0,1,0],[0,0,0,0,0,0],
    [0,0,0,1,0,0], [1,0,0,0,1,1], [0,0,0,0,1,0], [1,0,1,1,1,0], [0,1,0,1,1,1],[0,0,0,0,0,0],
    [1,0,0,0,1,0], [1,0,1,0,1,0], [0,1,0,1,0,0], [1,0,0,1,1,1], [1,0,1,0,1,0], [0,1,0,1,0,0], [0,1,0,0,1,1]]

#계좌 번호: 123-14-5678-900
e = [[0,0,0,1,0,0], [0,0,1,1,0,0], [0,0,0,1,0,1], [1,1,1,0,0,1],[0,0,0,0,0,0],
             [0,0,0,1,1,0], [0,1,1,1,1,1], [0,1,0,1,1,0], [1,0,1,0,0,1], [0,0,0,0,1,0], [0,1,0,0,0,0],[0,0,0,0,0,0],
             [0,0,1,1,1,1], [1,0,0,0,0,0], [1,1,0,0,0,0], [1,0,0,1,0,0], [0,0,1,0,0,1], [1,0,0,0,0,0], [1,0,0,1,1,0], [0,0,1,0,0,1], [1,0,0,0,1,0], [1,1,0,1,0,0], [1,1,0,1,1,0], [1,1,0,0,1,0], [0,0,1,0,0,1],
             [0,1,0,1,0,0], [0,1,0,1,1,0], [0,1,0,1,1,0]]

#그림을 그리고 있다.
f = [[0,0,0,1,0,0], [0,1,0,1,0,1], [0,0,0,0,1,0], [1,0,1,0,1,0], [0,1,0,0,0,1], [0,1,1,1,0,1],[0,0,0,0,0,0],
             [1,0,0,0,0,0], [1,0,1,0,0,1],[0,0,0,0,0,0],
             [1,0,1,0,1,0], [0,0,1,1,0,0], [0,1,0,1,0,0], [0,1,0,0,1,1]]

#젊은이는 나라의 기둥이다
g = [[0,0,0,1,0,1],[0,1,1,1,1,0],[0,1,0,0,0,1],[1,0,1,0,1,1],[1,0,1,0,1,0],
       [1,0,0,1,0,0],[1,0,1,0,1,1],[0,0,0,0,0,0],[1,0,0,1,0,0],[0,0,0,0,1,0],
       [1,1,0,0,0,1],[0,1,0,1,1,1],[0,0,0,0,0,0],[0,0,0,1,0,0],[1,0,1,0,1,0],
       [0,1,0,1,0,0],[1,0,1,1,0,0],[0,1,1,0,1,1],[1,0,1,0,1,0],[0,1,0,1,0,0],
       [0,1,0,0,1,1]]

#"여러분! 침착해야 합니다. '하늘이 무너져도 솟아날 구멍이 있다.'고 합니다."
h = [[0,1,1,0,0,1], [1,0,0,0,1,1], [0,0,0,0,1,0], [0,1,1,1,0,0], [0,0,0,1,1,0], [1,1,0,1,1,0], [0,1,1,0,1,0], [0,0,0,0,0,0],
     [0,0,0,0,1,1], [1,0,1,0,1,0], [0,1,0,0,0,1], [0,0,0,0,1,1], [1,1,0,0,0,1], [1,0,0,0,0,0], [0,1,0,1,1,0], [1,1,1,0,1,0], [0,0,1,1,1,0], [0,0,0,0,0,0],
     [0,1,0,1,1,0], [1,1,0,0,0,0], [1,0,0,1,0,0], [1,0,1,0,1,0], [0,1,0,1,0,0], [0,1,0,0,1,1], [0,0,0,0,0,0],
     [0,0,0,0,0,1], [0,1,1,0,0,1], [0,1,0,1,1,0], [1,0,0,1,0,0], [0,1,1,1,0,1], [1,0,1,0,1,0], [0,0,0,0,0,0],
     [1,0,0,0,1,0], [1,0,1,1,0,0], [1,0,0,1,0,0], [0,1,1,1,0,0], [0,0,0,1,0,1], [1,0,0,0,1,1], [0,1,0,1,0,0], [1,0,1,0,0,1], [0,0,0,0,0,0],
     [0,0,0,0,0,1], [1,0,1,0,0,1], [0,0,1,0,0,0], [1,1,0,0,0,1], [1,0,0,1,0,0], [0,1,0,0,0,0], [0,0,0,0,0,0],
     [0,0,0,1,0,0], [1,0,1,1,0,0], [1,0,0,0,1,0], [0,1,1,1,0,0], [0,1,1,0,1,1], [1,0,1,0,1,0], [0,0,0,0,0,0],
     [1,0,1,0,1,0], [0,0,1,1,0,0], [0,1,0,1,0,0], [0,1,0,0,1,1], [0,0,1,0,1,1], [0,0,1,0,0,0], [0,0,0,1,0,0], [1,0,1,0,0,1], [0,0,0,0,0,0],
     [0,1,0,1,1,0], [1,1,0,0,0,0], [1,0,0,1,0,0], [1,0,1,0,1,0], [0,1,0,1,0,0], [0,1,0,0,1,1], [0,0,1,0,1,1]]

#니체(독일의 철학자)는 이렇게 말했다.
i = [[1,0,0,1,0,0],[1,0,1,0,1,0],[0,0,0,0,1,1],[1,0,1,1,1,0],[0,1,1,0,0,1],[0,0,1,0,0,0],[0,1,0,1,0,0],[1,0,1,1,0,1],[1,0,1,0,1,0],[0,1,0,0,0,0],[0,1,0,1,1,1],[0,0,0,0,0,0],
     [0,0,0,0,1,1],[0,1,1,1,1,0],[0,1,0,1,1,0],[1,0,0,0,0,0],[0,0,0,1,0,1],[0,0,0,0,0,1],[0,0,1,0,1,1],[1,0,0,1,0,0],[1,0,1,0,1,1],[0,0,0,0,0,0],
     [1,0,1,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,0],[1,0,1,1,1,0],[0,0,0,0,0,0],
     [1,0,0,0,1,0],[0,1,0,0,0,0],[0,1,0,1,1,0],[1,1,1,0,1,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[0,1,0,0,1,1]]

#낱말[단어]
j = [[1,0,0,1,0,0], [0,1,1,0,0,1], [1,0,0,0,1,0], [0,1,0,0,0,0], [0,1,1,0,0,1], [0,1,1,0,0,0], [0,1,0,1,0,0], [0,1,0,0,1,0], [0,1,1,1,0,0], [0,0,0,0,1,1], [0,0,1,0,1,1]]

#예로부터 "민심은 천심이다."라고 하였다.
k = [[0,0,1,1,0,0], [0,0,0,0,1,0], [1,0,1,0,0,1], [0,0,0,1,1,0], [1,0,1,1,0,0], [1,1,0,0,1,0], [0,1,1,1,0,0], [0,0,0,0,0,0], [0,1,1,0,0,1], [1,0,0,0,1,0], [1,1,1,1,1,0], [0,0,0,0,0,1], [1,0,1,0,1,0], [0,1,0,0,0,1], [1,0,1,0,1,1], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,1,1,1,1,1], [0,0,0,0,0,1], [1,0,1,0,1,0], [0,1,0,0,0,1], [1,0,1,0,1,0], [0,1,0,1,0,0], [0,1,0,0,1,1], [0,0,1,0,1,1], [0,0,0,0,1,0], [1,1,0,0,0,1], [0,0,0,1,0,0], [1,0,1,0,0,1], [0,0,0,0,0,0], [0,1,0,1,1,0], [1,1,0,0,0,1], [1,0,0,0,1,1], [0,0,1,1,0,0], [0,1,0,1,0,0], [0,1,0,0,1,1]]

#일시: 2006년 2월 28일 13시
l = [[1,0,1,0,1,0], [0,1,0,0,0,0], [0,0,0,0,0,1], [1,0,1,0,1,0], [0,0,0,0,1,0], [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,1,1,1,1], [1,1,0,0,0,0], [0,1,0,1,1,0], [0,1,0,1,1,0], [1,1,0,1,0,0], [0,0,0,0,0,0], [1,0,0,1,0,0], [1,0,0,0,0,1], [0,0,0,0,0,0], [0,0,1,1,1,1], [1,1,0,0,0,0], [1,1,1,1,0,0], [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,1,1,1,1], [1,1,0,0,0,0], [1,1,0,0,1,0], [1,0,1,0,1,0], [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,1,1,1,1], [1,0,0,0,0,0], [1,0,0,1,0,0], [0,0,0,0,0,1], [1,0,1,0,1,0]]

#몫몫이[몽목씨]
m = [[1,0,0,0,1,0], [1,0,1,1,0,1], [0,0,1,0,0,0], [1,0,0,0,1,0], [1,0,1,1,0,1], [0,0,1,0,0,0], [1,0,1,0,1,0], [0,1,1,0,0,1], [0,1,1,0,0,0], [1,0,0,0,1,0], [1,1,1,1,1,1], [1,0,0,0,1,0], [1,0,1,1,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1], [1,0,1,0,1,0], [0,0,0,0,1,1], [0,0,1,0,1,1]]


for a in [a,b,c,d,e,f,g,h,i,j,k,l,m]:
    print("---")
    b = b2c.b2ccvt(a)
    print(b)
    c = b2h.b2hcvt(a)
    print(c)
    d = hbcvt.h2b.h2b(c)
    print(a)
    e = b2c.b2ccvt(d)
    print(e)
    f = b2h.b2hcvt(d)
    print(f)



