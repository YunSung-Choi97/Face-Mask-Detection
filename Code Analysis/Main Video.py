from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import winsound

# 네트워크 불러오기
# cv2.dnn.readNet(model, config, ... ) : 기존에 학습된 모델(네트워크)을 불러와 실행하는 함수
#    입력 : model(*.caffemodel 등), config(*.config 등), ... 등, model 외에는 생략가능
#    model : 훈련된 가중치를 저장하고 있는 이진 파일
#    config : 네트워크 구성을 저장하고 있는 텍스트 파일
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')


# Data Deep Learning.py 에서 학습한 model을 불러옴
model = load_model('8LBMI2.h5')


# 실시간 웹캠 읽기
# cv2.VideoCapture : 카메라를 통해 영상을 읽어옴. 연결된 카메라를 인덱스로 호출, VideoCapture타입
# .isOpened        : VideoCapture에 의해 정상적으로 Open되었는지 확인. bool타입
cap = cv2.VideoCapture(0)
while cap.isOpened():

    # ret : 비디오 프레임을 제대로 읽었는지 확인. bool타입. 
    # img : 읽어온 이미지(frame)
    # 비디오를 제대로 못 읽어왔다면 중단
    ret, img = cap.read()
    if not ret:
        break
    

    # h, w : 각각 읽어온 이미지의 세로길이, 가로길이. 픽셀수값, ex) h : 480, w : 640
    h, w = img.shape[:2]


    # 네트워크 입력 블롭(blob) 만들기
    # cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, ...)
    #     image : 입력 이미지
    #     scalefactor : 입력 영상 픽셀 값에 곱할 값. 기본값은 1
    #     size : 출력 영상의 크기. 기본값은 (0, 0)
    #     mean : 입력 영상 각 채널에서 뺄 평균 값. 기본값은 (0, 0, 0, 0)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(405, 405), mean=(104., 177., 123.))


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

        # 얼굴 영역 추출
        # 얼굴 정보 0 ~ 1로 정규화
        face = img[y1:y2, x1:x2]
        face = face/256

        # 얼굴이 화면 밖과 걸쳐있는 경우 인식하지 않음
        if (x2 >= w or y2 >= h):
            continue
        if (x1<=0 or y1<=0):
            continue
        
        # 추출한 이미지 크기변환
        face_input = cv2.resize(face,(200, 200))


        # face_input : 200 X 200 크기의 0 ~ 1로 표본화된 얼굴 이미지 정보 저장한 배열. shape = (1, 200, 200, 3)
        face_input = np.expand_dims(face_input, axis=0)
        face_input = np.array(face_input)


        # 얼굴 이미지의 마스크 착용 여부 확인
        # mask   : 마스크를 착용한 얼굴 이미지일 확률
        # nomask : 마스크를 미착용한 얼굴 이미지일 확률
        modelpredict = model.predict(face_input)
        mask=modelpredict[0][0]
        nomask=modelpredict[0][1]


        # 마스크를 착용한 얼굴로 판단한 경우
        if mask > nomask:
            # color : 초록색(BGR)
            # label : Mask (확률X100)
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        # 마스크를 미착용한 얼굴로 판단한 경우
        else:
            # color : 빨간색(BGR)
            # label : No Mask (확률X100)
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)

            #frequency = 2500  # Set Frequency To 2500 Hertz
            #duration = 1000  # Set Duration To 1000 ms == 1 second
            #winsound.Beep(frequency, duration)

        # 얼굴 영역 박스, 텍스트를 이미지에 추가
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)


    # 이미지 출력
    cv2.imshow('masktest',img)


    # escape 조건 (q키를 누를때까지 무한반복, 0이면 무한대기, waitkey : 특정시간 동안 대기하려면 ms값으로 입력)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break