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


# withimgnum    : 이미지(마스크 착용) 경로 디렉토리 내 파일 수
# withoutimgnum : 이미지(마스크 미착용) 경로 디렉토리 내 파일 수
# withimgnum, withoutimgnum 출력
withimgnum = len(os.listdir(with_dir))
withoutimgnum = len(os.listdir(without_dir))
print('total training withmask images:', withimgnum)
print('total training withoutmask images:', withoutimgnum)


# with_files    : 이미지(마스크 착용) 경로 디렉토리 내 파일명을 저장한 리스트
# without_files : 이미지(마스크 미착용) 경로 디렉토리 내 파일명을 저장한 리스트
with_files = os.listdir(with_dir)
without_files = os.listdir(without_dir)


for k in range(250,300):  # 반복. (파일 수 범위 내에서 반복가능)
    count=k  # 몇 번째 파일인지
    
    # img : 이미지 정보. 경로에서 파일을 읽어옴
    img = cv2.imread('raw_data/with_mask2/' + with_files[k])


    # h, w : 각각 읽어온 이미지의 세로길이, 가로길이. 픽셀수값, ex) h : 480, w : 640
    h, w = img.shape[:2]


    # 네트워크 입력 블롭(blob) 만들기
    # cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, ...)
    #     image : 입력 이미지
    #     scalefactor : 입력 영상 픽셀 값에 곱할 값. 기본값은 1
    #     size : 출력 영상의 크기. 기본값은 (0, 0)
    #     mean : 입력 영상 각 채널에서 뺄 평균 값. 기본값은 (0, 0, 0, 0)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(305,305), mean=(104., 177., 123.))

    # 이미지로 만든 blob 객체로 추론 진행
    # facenet : 얼굴을 찾는 모델(네트워크). 모델에 들어가는 input은 blob, 얼굴 영역 탐지 모델로 추론
    # dets    : facenet 순방향실행 결과를 추론하여 얼굴 영역 탐지결과 저장
    facenet.setInput(blob)
    dets = facenet.forward()


    # 마스크 착용여부 확인 (이미지 속 얼굴이 여러 개 있을 수 있으니 반복문 사용)
    for i in range(dets.shape[2]):

        # confidence : 검출한 결과 신뢰도. 낮을수록 얼굴 확률이 낮음
        # confidence < 0.5 인 경우는 얼굴로 인식하지 않음. threshold = 0.5로 지정
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue


        # 얼굴 바인딩 박스 찾기
        # x1, y1, x2, y2 : 얼굴 영역 바인딩 박스 네 꼭짓점
        # dets에서의 위치는 %로 검출. 따라서 w 또는 h 를 곱해줌
        x1 = int(dets[0, 0, i, 3] * w)  # 얼굴 좌측 좌표
        y1 = int(dets[0, 0, i, 4] * h)  # 얼굴 상단 좌표
        x2 = int(dets[0, 0, i, 5] * w)  # 얼굴 우측 좌표
        y2 = int(dets[0, 0, i, 6] * h)  # 얼굴 하단 좌표

        # 얼굴이 화면 밖과 걸쳐있는 경우 인식하지 않음
        if (x2 >= w or y2 >= h):
            continue

        # 얼굴 영역 추출
        face = img[y1:y2, x1:x2]

    
    # 추출한 이미지 크기변환
    face = cv2.resize(face, (200, 200))


    # file_name_path : 경로 + 파일명
    # 해당 경로에 파일명으로 이미지 저장
    file_name_path = 'train_data2/mask/trainnm' + str(count) + '.jpg'
    cv2.imwrite(file_name_path, face)

    
    print(count)

print("CopyComplete")