from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import winsound

# 네트워크 불러오기
# cv2,dnn.readNet(model, config=None, ...)
# model : 훈련된 가중치를 저장하고 있는 이진 파일 이름
# config : 네트워크 구성을 저장하고 있는 텍스트 파일 이름
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# model을 제외한 다른 정보는 없을 수도 있음.
# 딥러닝 프레임워크가 카페일 경우 model 확장자 .caffemodel, config 확장자 .prototxt


model = load_model('8LBMI2.h5')

# 실시간 웹캠 읽기

# 첫번째(0번째) 카메라를 통해 VideoCapture타입의 객체로 읽어옴
cap = cv2.VideoCapture(0)
i = 0

while cap.isOpened():

    # ret : bool타입. 비디오 프레임을 제대로 읽었는지
    # img : 읽어온 이미지(frame)
    ret, img = cap.read()
    if not ret:
        break

    # 일반적으로 width = int(cap.get(3)), height = int(cap.get(4))로
    h, w = img.shape[:2]  # h : 480, h : 640

    # 네트워크 입력 블롭(blob) 만들기
    # cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, ...)
    # image : 입력 영상
    # scalefactor : 입력 영상 픽셀 값에 곱할 값. 기본값은 1
    # size : 출력 영상의 크기. 기본값은 (0, 0)
    # mean : 입력 영상 각 채널에서 뺄 평균 값. 기본값은 (0, 0, 0, 0)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(405, 405), mean=(104., 177., 123.))

    # https://thebook.io/006939/ch16/01/02-06/

    # 네트워크 입력 설정하기
    # readNet으로 만든 객체에 blob 설정
    facenet.setInput(blob)

    # 네트워크 순방향 실행(추론)
    # 추론을 진행할 때 이용. 출력 레이어 이름 지정을 할 수 있음
    # 출력 : 지정한 레이어의 출력 블롭
    dets = facenet.forward()


    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w)  # 얼굴 좌측 좌표
        y1 = int(dets[0, 0, i, 4] * h)  # 얼굴 상단 좌표
        x2 = int(dets[0, 0, i, 5] * w)  # 얼굴 우측 좌표
        y2 = int(dets[0, 0, i, 6] * h)  # 얼굴 하단 좌표

        face = img[y1:y2, x1:x2]
        face = face/256

        if (x2 >= w or y2 >= h):
            continue
        if (x1<=0 or y1<=0):
            continue

        face_input = cv2.resize(face,(200, 200))

        # ?
        face_input = np.expand_dims(face_input, axis=0)
        face_input = np.array(face_input)

        modelpredict = model.predict(face_input)
        mask=modelpredict[0][0]
        nomask=modelpredict[0][1]

        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            #frequency = 2500  # Set Frequency To 2500 Hertz
            #duration = 1000  # Set Duration To 1000 ms == 1 second
            #winsound.Beep(frequency, duration)

        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('masktest',img)

    # escape 조건 (q키를 누를때까지 무한반복, 0이면 무한대기, waitkey : 특정시간 동안 대기하려면 ms값으로 입력)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break