import os, re, glob
import cv2
import numpy as np
import shutil
from keras.models import load_model

# 입력 : 이미지 경로(상대/절대경로)
# 이미지 파일을 flag값에 따라서 읽어들임. (여기서 flag 생략)
# 출력 : 이미지 객체 행렬 / 256. numpy.ndarray 자료형
def Dataization(img_path):
    img = cv2.imread(img_path)
    return (img / 256)


src = []    # 이미지 경로+파일이름
name = []   # 이미지 파일이름
test = []   # 이미지 객체행렬/256

image_dir = 'test_data/'
# image_dir 경로 내의 모든 파일과 디렉토리의 리스트를 리턴
for file in os.listdir(image_dir):
    # .jpg 파일이라면
    if (file.find('.jpg') is not -1):
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))

test = np.array(test)
print(test.shape)   # 이미지는 3차원 행렬 나옴. (Y축, X축, 3(BGR))
model = load_model('6LBMIv2-20.h5')
predict = model.predict(test)
print(predict.shape)
print("ImageName : , Predict : [mask, nomask]")
for i in range(len(test)):
    print(name[i] + " : , Predict : " + str(predict[i]))