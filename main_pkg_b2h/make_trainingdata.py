
# 트레이닝 데이터 만들기
# 카테고리별로 2000개씩 이미지 데이터 만들기

import random
from PIL import Image

# 만약 seg_path 폴더에 파일이 있으면 모두 삭제
import os
def removeAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)

# background = 'a'

#### 모델링 코드랑 짬뽕되어 있어서 조금 헷갈릴 수도~~ 경로는 알아서 보자...

# dotdot 폴더 : 점 하나 찍힌 이미지들 모음
target_dir = 'C:/YC/project01/project3/dotdot/'
a = target_dir + background +'.jpg'

bfile = Image.open(a)

bx = bfile.size[0]
by = bfile.size[1]
dx = round(bx/2)
dy = round(by/3)
print(bx, by)
print(dx, dy)

bbb = [[0,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,0,1,1],
       [0,0,0,1,0,0],[0,0,0,1,0,1],[0,0,0,1,1,0],[0,0,0,1,1,1],
       [0,0,1,0,0,0],[0,0,1,0,0,1],[0,0,1,0,1,0],[0,0,1,0,1,1],
       [0,0,1,1,0,0],[0,0,1,1,0,1],[0,0,1,1,1,0],[0,0,1,1,1,1],
       [0,1,0,0,0,0],[0,1,0,0,0,1],[0,1,0,0,1,0],[0,1,0,0,1,1],
       [0,1,0,1,0,0],[0,1,0,1,0,1],[0,1,0,1,1,0],[0,1,0,1,1,1],
       [0,1,1,0,0,0],[0,1,1,0,0,1],[0,1,1,0,1,0],[0,1,1,0,1,1],
       [0,1,1,1,0,0],[0,1,1,1,0,1],[0,1,1,1,1,0],[0,1,1,1,1,1],
       [1,0,0,0,0,0],[1,0,0,0,0,1],[1,0,0,0,1,0],[1,0,0,0,1,1],
       [1,0,0,1,0,0],[1,0,0,1,0,1],[1,0,0,1,1,0],[1,0,0,1,1,1],
       [1,0,1,0,0,0],[1,0,1,0,0,1],[1,0,1,0,1,0],[1,0,1,0,1,1],
       [1,0,1,1,0,0],[1,0,1,1,0,1],[1,0,1,1,1,0],[1,0,1,1,1,1],
       [1,1,0,0,0,0],[1,1,0,0,0,1],[1,1,0,0,1,0],[1,1,0,0,1,1],
       [1,1,0,1,0,0],[1,1,0,1,0,1],[1,1,0,1,1,0],[1,1,0,1,1,1],
       [1,1,1,0,0,0],[1,1,1,0,0,1],[1,1,1,0,1,0],[1,1,1,0,1,1],
       [1,1,1,1,0,0],[1,1,1,1,0,1],[1,1,1,1,1,0],[1,1,1,1,1,1]
]

jcnt = 2000        #폴더당 만들 점자파일 개수
dotv = 49        #랜덤 닷 종류? 개수?

for bb in bbb:
    print(bb)
    ##폴더 비우고 시작하자
    # newname = 'd' + str(bb).replace("[", "").replace("]", "").replace(",", "").replace(" ", "")
    # removeAllFile(target_dir + 'train/' + newname)
    for j in range(1,jcnt+1):
        newb = Image.new("RGB", (1954, 3072), (256,256,256))

        icnt = 0
        for i in bb:
            icnt += 1
            a1 = 'a' + str(random.randrange(1,dotv))
            a1 = target_dir + a1 + '.jpg'
            file1 = Image.open(a1)

            file1 = file1.resize((dx, dy))

            if i == 1:
                if icnt == 1:
                    area = (0,0,dx,dy)
                elif icnt == 2:
                    area = (0,dy,dx,dy*2)
                elif icnt == 3:
                    area = (0,dy*2,dx,dy*3)
                elif icnt == 4:
                    area = (dx,0,dx*2,dy)
                elif icnt == 5:
                    area = (dx,dy,dx*2,dy*2)
                elif icnt == 6:
                    area = (dx,dy*2,dx*2,dy*3)

                newb.paste(file1, area)
        # newb.show()
        newname = 'd' + str(bb).replace("[","").replace("]","").replace(",","").replace(" ","")
        print(newname)
        newb = newb.resize((70, 110))
        newb.save(target_dir +'train/'+newname+'/'+ '0618_'+ newname + '_' + str(j)+'.jpg')
