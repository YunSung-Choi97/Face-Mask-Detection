import dlib
import cv2
import numpy as np
from PIL import Image, ImageFile
import os

# detector  : 정면얼굴 검출기
# predictor : 안면 랜드마킹 학습모델 데이터. (dlib에서 제공된 "shape_predictor_68_face_landmarks.dat" 이용)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# without_dir   : 경로명
# withoutimgnum : 해당 경로 디렉토리 내의 파일 수
# without_files : 해당 경로 디렉토리 내의 파일명
without_dir = os.path.join('raw_data/without_mask1')
withoutimgnum = len(os.listdir(without_dir))
without_files = os.listdir(without_dir)


# 해당 경로 디렉토리 내의 파일 수 출력
print('total training withoutmask images:', len(os.listdir(without_dir)))


# 이미지(마스크 미착용)에 대하여 적절한 마스크 이미지를 합성. (파일 수만큼 반복)
# 1) 이미지 읽어오기
# 2) 읽어온 이미지 속 안면 랜드마킹
# 3) 마스크 이미지 읽어오기
# 4) 이미지(마스크 미착용)내에 있는 얼굴에 맞는 마스크 이미지 제작
# 5) 마스크 이미지를 합성할 위치 계산
# 6) 마스크 이미지 합성
for k in range(0,623):
    count = k  # 몇 번째 파일인지

    # 1) 이미지 읽어오기

    # img : 이미지 정보. 경로에서 파일을 읽어옴
    # rows, cols : 이미지 세로, 가로. (numpy)shape으로 배열정보 받아옴. img.shape : (세로, 가로, BGR).
    # rects : img에서 검출한 얼굴 정보. ex) rectangles[[(43, 118) (266, 341)]] : 사각형의 (얼굴)좌상단 후하단 값
    img = cv2.imread('raw_data/without_mask1/' + without_files[k], 1)
    rows, cols = img.shape[:2]
    rects = detector(img, 1)
    

    # 2) 읽어온 이미지 속 안면 랜드마킹

    # rects에서 인덱스(i)와 값(rect)을 함께 불러옴. ex) rect : rectangle(43, 118, 277, 341)
    for i, rect in enumerate(rects):
        
        # shape : 이미지(마스크 미착용)에서 안면 랜드마킹한 값을 리턴.
        shape = predictor(img, rect)
        
        # "shape_predictor_68_face_landmarks.dat"이 68개의 점을 통해 안면 랜드마킹 하기때문에 68번 반복
        for j in range(68):
            
            # x, y : 마스크를 씌울 좌표
            # color : 초록색 (BGR)
            x, y = shape.part(j).x, shape.part(j).y
            color = (0, 255, 0)

            
            # 3번(왼쪽 턱), 8번(아래 턱), 13번(오른쪽 턱), 29번(코의 중심)일 경우에
            # color를 빨간색으로 변경
            # 각각의 좌표 저장. ex) left : array[77, 269]
            if (j == 3):
                color = (0, 0, 255)
                left = np.array([x, y])
            elif (j == 8):
                color = (0, 0, 255)
                chin = np.array([x, y])
            elif (j == 13):
                color = (0, 0, 255)
                right = np.array([x, y])
            elif (j == 29):
                color = (0, 0, 255)
                nose = np.array([x, y])
            
            # cv2.putText : 이미지에 글자 삽입함수. 입력 : 이미지, 입력할 문자, 좌표, 폰트, 사이즈, 색상
            # 이미지(마스크 미착용)에 안면 랜드마크 지점들을 초록색 숫자로 표시 (마스크 씌울 위치는 빨간색 숫자)
            cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, color)


    # 점과 선(두 점을 잇는) 사이의 거리 반환.
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


    # 3) 마스크 이미지 읽어오기

    # mask_img      : 이미지(마스크)파일 불러옴
    # width, height : 각각 mask_img의 가로, 세로 ex) 234, 146
    mask_img = Image.open('blue-mask.png')
    width = mask_img.width
    height = mask_img.height

    # width_ratio   : 얼굴의 가로길이와 마스크의 가로길이 사이의 비율 설정
    # new_height    : 얼굴에 적절한 마스크 세로길이.
    width_ratio = 1.2
    new_height = int(np.linalg.norm(left - right))

    # ???
    # new_height : 왼쪽 턱과 오른쪽 턱 사이의 가로길이
    # left : 77, 269 / right : 249, 279 >> 172
    # left : 75, 277 / right : 262, 279 >> 187
    # left : 55, 275 / right : 244, 276 >> 189


    # 4) 이미지(마스크 미착용)내에 있는 얼굴에 맞는 마스크 이미지 제작
    
    # 4-1) 왼쪽 마스크 이미지 설정
    # mask_left_img     : 이미지(마스크)의 왼쪽 절반
    # mask_left_width   : 얼굴의 왼쪽 가로길이. 왼쪽 턱 ~ (코의 중심--아래턱)까지의 거리
    #                   : 마스크의 왼쪽 가로길이. 얼굴의 왼쪽 가로길이 * width_ratio(마스크:얼굴 비율)
    # mask_left_img의 크기변경.
    mask_left_img = mask_img.crop((0, 0, width // 2, height))
    mask_left_width = get_distance_from_point_to_line(left, nose, chin)
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # 4-2) 오른쪽 마스크 이미지 설정
    # mask_right_img    : 이미지(마스크)의 오른쪽 절반
    # mask_right_width  : 얼굴의 오른쪽 가로길이. 오른쪽 턱 ~ (코의 중심--아래턱)까지의 거리
    #                   : 마스크의 오른쪽 가로길이. 얼굴의 오른쪽 가로길이 * width_ratio(마스크:얼굴 비율)
    # mask_right_img의 크기변경.
    mask_right_img = mask_img.crop((width // 2, 0, width, height))
    mask_right_width = get_distance_from_point_to_line(right, nose, chin)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # 4-3) 전체 마스크 이미지 제작
    # size      : 마스크 이미지 크기.
    # mask_img  : 마스크 이미지. 왼쪽 마스크 이미지 + 오른쪽 마스크 이미지
    # Image.new : 새로운 이미지 제작함수. 입력 : RGBA(rgb + alpha(투명도)) 모드, 크기. + 바탕색상을 입력받을 수 있다.(default : (0, 0, 0))
    # .paste    : 이미지 덧씌우는 함수. 입력 : 덧씌울 이미지, 덧씌울 이미지 위치, 덧씌울 이미지 크기
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA', size)
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    # 4-4) 얼굴과 마스크 수평 맞추기
    # angle             : 얼굴이 기울어진 각도
    # rotated_mask_img  : 얼굴이 기울어진 각도만큼 회전된 마스크 이미지
    # .arctan2          : arctan계산. .arctan과 달리 (-PI ~ PI) 값 반환
    # .rotate           : 이미지 회전함수.
    angle = np.arctan2(chin[1] - nose[1], chin[0] - nose[0])
    rotated_mask_img = mask_img.rotate(angle, expand=True)


    # 5) 마스크 이미지를 배치할 위치 계산
    
    # 5-1) 마스크 이미지를 놓을 중심점 찾기
    # center_x : 코의 중심과 아래 턱의 가로 중점 좌표
    # center_y : 코의 중심과 아래 턱의 세로 중점 좌표
    center_x = (nose[0] + chin[0]) // 2
    center_y = (nose[1] + chin[1]) // 2
    
    
    # 5-2) 비대칭과 회전을 고려한 마스크 배치할 위치 계산
    # offset : 비대칭 마스크를 적절히 배치하기 위한 offset. 전체 마스크 이미지 가로길이의 절반 - 왼쪽 마스크 이미지 가로길이
    # radian : 얼굴이 기울어진 각도(radian)
    # box_x  : 마스크 이미지를 배치할 x 좌표
    # box_y  : 마스크 이미지를 배치할 y 좌표
    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2


    # 6) 마스크 이미지 합성
    # img2          : 이미지(마스크 미착용)파일 불러옴
    # 이미지(마스크 미착용)에 제작한 마스크 이미지 덧씌우기
    # file_name_path : 경로 + 파일명
    # 해당 경로에 마스크 이미지가 덧씌워진 이미지 파일 저장
    img2 = Image.open('raw_data/without_mask1/' + without_files[k])
    img2.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)
    file_name_path = 'test_data_final/blue-mask/testbluem' + str(count) + '.jpg'
    img2.save(file_name_path)

    print(count)