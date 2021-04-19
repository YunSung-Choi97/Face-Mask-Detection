from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.models import load_model
import os, re, glob
import cv2
import numpy as np


# 로컬 이미지 하위 폴더에 카테고리 생성
# groups_folder_path : 학습 데이터가 카테고리로 분류된 경로. 학습할 데이터의 상위 경로
# categories         : 학습 데이터 카테고리. 폴더명
# num_classes        : 학습 데이터 카테고리의 수
groups_folder_path = 'train_data_2(recommand)'
categories = ['mask','nomask']
num_classes = len(categories)


# X : 0 ~ 1 사이값으로 정규화시켜 이미지 정보를 저장할 리스트
# Y : X에 저장된 이미지의 Label 정보를 저장할 리스트
X = []
Y = []

# 카테고리의 데이터를 바로 머신러닝에 사용할 수 있는 데이터 형태로 바꾸기
# category별로 이미지 data와 label matching. label이란 data에 붙은 정답

# categories에서 인덱스(idex)와 값(category)을 함께 불러옴
for idex, categorie in enumerate(categories):

    # label     : 리스트 (크기 = 카테고리 수, 값 = 0 (카테고리에 해당하는 인덱스 값만 1)
    # image_dir : 이미지 경로
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path+'/'+ categorie + '/'


    # 해당 경로의 (하위 디렉토리를 포함) 모든 파일명을 filename으로 전달. (하위 디렉토리 속 파일은 파일까지의 경로를 포함한 파일명)
    for top, dir, f in os.walk(image_dir):
        for filename in f:

            # img : 이미지 정보. 경로에서 파일을 읽어옴
            # X에 0~1로 정규화된 이미지 정보 저장
            # Y에 이미지에 알맞은 label 저장
            img = cv2.imread(image_dir + filename)
            X.append(img / 256)
            Y.append(label)

# 리스트(X, Y)를 numpy에서 활용할 수 있는 형태의 배열형태로 변환
Xtr = np.array(X)
Ytr = np.array(Y)
X_train, Y_train = Xtr, Ytr

# ???
# Xtr과 Ytr을 굳이 X_train과 Y_train으로 바꾼 이유?? 복사한 이유?? 굳이??


# X_train 출력          : (800, 200, 200, 3)
# X_train[0]을 빼고 출력 : (200, 200, 3)
print(X_train.shape)
print(X_train.shape[1:])


# ppt 자료 참고
# Convolution Layer
# Pooling Layer
# Dropout Layer (테스트 단계에서는 어떤 유닛도 드랍하지 않음. 훈련 단계에서만 dropout으로 출력값 줄임)
# Flatten Layer (1차원 배열로 바꾸어주는 역할을 수행. Fully connected network로 연산)
# Dense Layer (Last Layer의 출력은 mask / no mask 둘 중 하나로 나와야 하기때문에 unit 2)


# model에 Sequential 방식으로 layer 추가
model = Sequential()

# Convolution layer (channel = 16, filter = 3X3)
# Pooling layer (Max pooling, filter = 2X2)
# Dropout layer (dropout ration = 0.25)
model.add(Conv2D(16,3,3, border_mode='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolution layer (channel = 20, filter = 3X3, ReLU function)
# Pooling layer (Max pooling, filter = 2X2)
# Dropout layer (dropout ration = 0.25)
model.add(Convolution2D(20, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolution layer (channel = 64, filter = 3X3, ReLU function)
# Pooling layer (Max pooling, filter = 2X2)
# Dropout layer (dropout ration = 0.25)
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolution layer (channel = 64, filter = 3X3, ReLU function)
# Pooling layer (Max pooling, filter = 2X2)
# Dropout layer (dropout ration = 0.25)
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolution layer (channel = 64, filter = 3X3, ReLU function)
# Pooling layer (Max pooling, filter = 2X2)
# Dropout layer (dropout ration = 0.25)
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# Classifier

# Flatten Layer
# Dense Layer (unit = 200, ReLU function)
# Dense layer (unit = 2, Softmax function)
model.add(Flatten())
model.add(Dense(200, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

# model 컴파일 (손실함수 = binary_crossentropy, 옵티마이저 = adam optimizer, 척도 = accuracy)
# model 학습 (batch size = 40, epochs = 20)
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
hist=model.fit(X_train, Y_train, batch_size=40, nb_epoch=20)


# 학습을 완료한 model 저장
model.save('8LBMI3.h5')