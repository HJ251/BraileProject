
from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split
import cv2

# 미리 만들어 놔야 하는 폴더 목록
# 이미지 폴더 + 이미지 : ../img/dotimg
# 모델 폴더 : ../model
# 넘파이 데이터 배열 폴더 : ../numpy_data


# caltech_dir = "../img/dotimg/train"
# categories = ["d100000", "d010000", "d110000", "d001000", "d101000", "d011000", \
#               "d111000", "d000100", "d100100", "d010100", "d110100", "d001100",\
#                "d101100", "d011100", "d111100", "d000010", "d100010", "d010010", \
#               "d110010", "d001010", "d101010", "d011010", "d111010", "d000110",\
#                "d100110", "d010110", "d110110", "d001110","d101110","d011110",\
#               "d111110","d000001","d101001","d010001","d110001","d001001","d101001",\
#               "d011001","d111001","d000101","d100101","d010101","d110101","d001101",\
#               "d101101","d011101","d111101","d000011","d100011","d010011",\
#               "d110011","d001011","d101011","d011011","d111011","d000111","d100111",\
#               "d010111","d110111","d001111","d101111", "d011111","d111111" ]
# nb_classes = len(categories)
#
# image_w = 70
# image_h = 110
#
# pixels = image_h * image_w * 3
#
# X = []
# y = []
#
# for idx, cat in enumerate(categories):      # idx:인덱스 순서, cat:값
#
#     # one-hot 돌리기.
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1
#
#     image_dir = caltech_dir + "/" + cat
#     # files = glob.glob(image_dir + "/*.png")
#     files = glob.glob(image_dir + "/*.jpg")     # "chou_giuck" "dga", "eee", "ea"는 jpg 파일이다
#     print(cat, " 파일 길이 : ", len(files))
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)
#
#         X.append(data)
#         y.append(label)
#
#         # 잘 돌아가나 확인 용도
#         # if i % 700 == 0:
#         #     print(cat, " : ", f)
#
# X = np.array(X)
# y = np.array(y)
# # 1 0 0 0 이면 airplanes
# # 0 1 0 0 이면 buddha 이런식
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# xy = (X_train, X_test, y_train, y_test)
# np.save("../numpy_data/multi_image_data.npy", xy)
#
# print("ok", len(y))
#
# import os, glob, numpy as np
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import matplotlib.pyplot as plt
# import keras.backend.tensorflow_backend as K
#
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#
# X_train, X_test, y_train, y_test = np.load('../numpy_data/multi_image_data.npy', allow_pickle = True)
# print(X_train.shape)
# # # X_train = X_train.reshape()
# # # 차원 늘리는 것의 np.expand_dims의 반대 컨셉 찾아보기
# print(X_train.shape[0])

categories = ["d100000", "d010000", "d110000", "d001000", "d101000", "d011000", \
              "d111000", "d000100", "d100100", "d010100", "d110100", "d001100",\
               "d101100", "d011100", "d111100", "d000010", "d100010", "d010010", \
              "d110010", "d001010", "d101010", "d011010", "d111010", "d000110",\
               "d100110", "d010110", "d110110", "d001110","d101110","d011110",\
              "d111110","d000001","d101001","d010001","d110001","d001001","d101001",\
              "d011001","d111001","d000101","d100101","d010101","d110101","d001101",\
              "d101101","d011101","d111101","d000011","d100011","d010011",\
              "d110011","d001011","d101011","d011011","d111011","d000111","d100111",\
              "d010111","d110111","d001111","d101111", "d011111","d111111" ]
# nb_classes = len(categories)
#
# #일반화
# X_train = X_train.astype(float) / 255
# X_test = X_test.astype(float) / 255
#
# # -------------------------------------------------
# #                  모델링 코드
# # -------------------------------------------------
# with K.tf_ops.device('/device:GPU:0'):
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     # CNN
#     # 여기 레이어를 더 넣던지 해서 성능을 높히자
#     model.add(Flatten())
#
#     # # case1 : 기존코드
#     # model.add(Dense(256, activation='relu'))
#     # model.add(Dropout(0.5))
#     # model.add(Dense(nb_classes, activation='softmax'))
#
#     # case2
#     model.add(Dense(units=512, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=512, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=nb_classes, activation='softmax'))
#
#     # # case3
#     # model.add(Dense(units=256, activation='relu'))
#     # model.add(Dropout(0.3))
#     # model.add(Dense(units=256, activation='relu'))
#     # model.add(Dropout(0.3))
#     # model.add(Dense(units=nb_classes, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model_dir = '../model'
#
# # -------------------------------------------------
#
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#
#     model_path = model_dir + '/multi_img_classification2.model'
#     checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
#     early_stopping = EarlyStopping(monitor='val_loss', patience=6)
#
#
#
# model.summary()
#
# history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
#
# print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
#
# y_vloss = history.history['val_loss']
# y_loss = history.history['loss']
#
# x_len = np.arange(len(y_loss))
#
# plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
# plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
# plt.legend()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()




# 테스트

from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import time

def model_test(caltech_dir="./img/dotimg/test3"):
    # caltech_dir = "../img/dotimg/test3"

    # 모델2
    # image_w = 70
    # image_h = 110

    # 모델3
    image_w = 35
    image_h = 55

    pixels = image_h * image_w * 3

    X = []
    filenames = []
    files = glob.glob(caltech_dir+"/*.*")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))

        # 리사이징 확인
        # cv2.imshow('img', img)
        # plt.imshow(img)
        # plt.show()

        # time.sleep(1)

        data = np.asarray(img)
        filenames.append(f)
        X.append(data)

    print(filenames)

    # # filenames를 파일 이름순으로 버블 정렬 해주자
    # # li : 숫자 리스트
    # def bubble_sort(li):
    #     length = len(li) - 1
    #     for i in range(length):
    #         for j in range(length - i):
    #             a = li[j].rfind('_')
    #             file_num1 = int(li[j][a + 1:-4])
    #
    #             b = li[j + 1].rfind('_')
    #             file_num2 = int(li[j + 1][a + 1:-4])
    #
    #             if file_num1 > file_num2:
    #                 li[j], li[j + 1] = li[j + 1], li[j]
    #
    # bubble_sort(filenames)
    #
    # print(filenames)

    X = np.array(X)
    model = load_model('./model/multi_img_classification3.model')

    prediction = model.predict(X)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    cnt = 0

    suc = 0
    total = 0
    result_dot = []

    for i in prediction:
        pre_ans = i.argmax()  # 예측 레이블
        # print(i)
        # print(pre_ans)
        pre_ans_str = ''

        for idx in range(len(categories)):
            if pre_ans == idx: pre_ans_str = categories[idx]

        # 01 하나 담을 리스트
        imsi = []
        for idx in range(len(categories)):
            if i[idx] >= 0.8 :
                print("해당 "+filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
                a = 'd' + filenames[cnt].split("\\")[1][:6]
                total += 1
                # if a == pre_ans_str:
                #     print('성공!')
                #     suc += 1
                # else:
                #     print('실패!')

                one = dot01 = pre_ans_str[1:]
                for item in one:
                    imsi.append(int(item))

                result_dot.append(imsi)

        cnt += 1

    print('-'*40)
    print('총 테스트 개수 : ', total)
    # print('총 성공 개수 : ', suc)

    return result_dot


