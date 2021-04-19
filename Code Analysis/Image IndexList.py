import cv2
import numpy as np
import os

# 네트워크 불러오기
# cv2.dnn.readNet   : 기존에 학습된 모델(네트워크)을 불러와 실행하는 함수
#                   입력 : model(*.caffemodel 등), config(*.config 등), ... 등 
#                   model : 훈련된 가중치를 저장하고 있는 이진 파일
#                   config : 네트워크 구성을 저장하고 있는 텍스트 파일
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')


# with_dir    : 이미지(마스크 착용) 경로
# without_dir : 이미지(마스크 미착용) 경로
with_dir = os.path.join('raw_data/with_mask2')
without_dir = os.path.join('raw_data/without_mask1')


# 이미지(마스크 착용) 경로 디렉토리 내 파일 수 출력
# 이미지(마스크 미착용) 경로 디렉토리 내 파일 수 출력
print('total training withmask images:', len(os.listdir(with_dir)))
print('total training withoutmask images:', len(os.listdir(without_dir)))


# withimgnum    : 이미지(마스크 착용) 경로 디렉토리 내 파일 수
# withoutimgnum : 이미지(마스크 미착용) 경로 디렉토리 내 파일 수
withimgnum = len(os.listdir(with_dir))
withoutimgnum = len(os.listdir(without_dir))


# with_files    : 이미지(마스크 착용) 경로 디렉토리 내 파일명을 저장한 리스트
# without_files : 이미지(마스크 미착용) 경로 디렉토리 내 파일명을 저장한 리스트
with_files = os.listdir(with_dir)
without_files = os.listdir(without_dir)


# with_files에서 인덱스(0, 1, ...)와 값(파일명1, 파일명2, ...)을 함께 불러와 출력
for i in enumerate(with_files):
    print(i)