
import glob

from main_pkg_b2h.dot_noising import *

imgpath_origin = '../img/noiseTest/'
dstpath = '../img/noiseTest/result/'

# a = imgpath.rfind('/')
# filename = imgpath[a+1:-4]

# ---------------------------------

files = glob.glob(imgpath_origin + "*.jpg")     # 모든 jpg 파일을 가져와라

print("노이즈 제거할 파일 개수 : ", len(files))
# print(files)


for i, f in enumerate(files):
    index = f.find('\\', 4)
    f = f[:index] + '/' + f[index+1:]
    # print(f)

    a = f.rfind('/')
    filename = f[a+1:-4]
    # print(filename)

    imgpath = f

    # ---------------------------------

    for idx in range(5):
        dst = dstpath + filename + '_result' + str(idx+1) + '.jpg'
        if idx == 0:
            noising01(imgpath, dst)
        elif idx == 1:
            noising02(imgpath, dst)
        elif idx == 2:
            noising03(imgpath, dst)
        elif idx == 3:
            noising04(imgpath, dst)
        else:
            noising05(imgpath, dst)

print('-finished-')

