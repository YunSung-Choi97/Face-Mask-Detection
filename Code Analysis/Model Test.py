import os, re, glob
import cv2
import numpy as np
import shutil
from keras.models import load_model


# 이미지 정보를 0 ~ 1 사이로 정규화하는 함수
def Dataization(img_path):
    img = cv2.imread(img_path)
    return (img / 256)


src = []    # 이미지 경로+파일이름
name = []   # 이미지 파일이름
test = []   # 정규화된 이미지 정보


# image_dir : 이미지 경로
image_dir = 'test_data/'

# image_dir 경로 디렉토리 내 모든 파일명, 디렉토리명에 대하여
for file in os.listdir(image_dir):
    # .jpg 파일이라면
    if (file.find('.jpg') is not -1):

        # src에 저장. 이미지 경로+파일이름
        # name에 저장. 이미지 파일이름
        # test에 저장. 정규화된 이미지 정보
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))


# 이미지 정보를 numpy에서 활용할 수 있는 형태의 배열형태로 변환
test = np.array(test)

# 학습된 model 불러옴
model = load_model('8LBMI2.h5')

# 이미지에 대하여 마스크 착용 여부 판별
predict = model.predict(test)

# test.shape 출력. (파일수, Y축, X축, 3) ex) (197, 200, 200, 3)
# predict.shape 출력. (파일수, 학습 category수) ex) (197, 2)
# 파일명, [mask 착용 확률, mask 미착용 확률] 출력
print(test.shape)
print(predict.shape)
print("ImageName : , Predict : [mask, nomask]")
for i in range(len(test)):
    print(name[i] + " : , Predict : " + str(predict[i]))