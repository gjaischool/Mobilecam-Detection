# Mobilecam-Detection
# 사이즈 224*224 수정
import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance as dist

# Pygame 초기화
pygame.mixer.init()

# 상황별 알림음 파일 경로 설정
drowsiness_sound_path = 'siren-alert-96052.mp3'
eye_closed_sound_path = 'warning-alert-this-is-not-a-test-141753.mp3'  # 추가된 눈 감음 경고음

# 핸드폰 스트리밍 URL 설정 (IP 웹캠 앱에서 확인한 URL)
stream_url = 'http://192/video'

# 임계값 설정
EAR_THRESHOLD = 0.25  
PITCH_THRESHOLD = 8   
FRAME_THRESHOLD = int(30 * 0.8)  
YAWN_THRESHOLD = 1.5  
YAWN_FREQUENCY_THRESHOLD = 3  

# 얼굴 인식 타원 틀 크기 설정
FACE_BOX_WIDTH = 250
FACE_BOX_HEIGHT = 300

# EAR 계산 함수 (눈 감김 비율)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# MAR 계산 함수 (하품 비율)
def calculate_mar(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# 오일러 각도 계산 함수 (Yaw, Pitch, Roll)
def get_euler_angles(shape, frame):
    image_points = np.array([
        shape[30],  
        shape[8],   
        shape[36],  
        shape[45],  
        shape[48],  
        shape[54]   
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             
        (0.0, -330.0, -65.0),        
        (-225.0, 170.0, -135.0),     
        (225.0, 170.0, -135.0),      
        (-150.0, -150.0, -125.0),    
        (150.0, -150.0, -125.0)      
    ])

    focal_length = frame.shape[1]
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    yaw, pitch, roll = angles
    return yaw, pitch, roll

# 얼굴 탐지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/USER/Desktop/shape_predictor_68_face_landmarks.dat')

# 눈과 입 랜드마크 인덱스
(lStart, lEnd) = (36, 41)
(rStart, rEnd) = (42, 47)
(mStart, mEnd) = (48, 67)

# 실시간 스트리밍 처리 함수
def process_stream():
    cap = cv2.VideoCapture(stream_url)
    eye_blink_counter = 0
    yawn_counter = 0
    head_tilt_counter = 0
    alarm_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("스트리밍 연결에 문제가 발생했습니다.")
            break

        # 1. Grayscale 모드로 프레임을 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Grayscale을 BGR 형식으로 변경

        # 2. 프레임 크기를 224x224로 조정
        frame = cv2.resize(frame, (224, 224))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # 얼굴 맞춤 타원 틀 설정
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2

        # 타원 틀의 마스크 생성 (타원 내부는 그대로, 외부는 블러 처리)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (FACE_BOX_WIDTH // 2, FACE_BOX_HEIGHT // 2), 0, 0, 360, 255, -1)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 블러 처리한 배경과 얼굴 영역 결합
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        inverse_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=inverse_mask)
        frame = cv2.add(masked_frame, background)

        cv2.ellipse(frame, (center_x, center_y), (FACE_BOX_WIDTH // 2, FACE_BOX_HEIGHT // 2), 0, 0, 360, (0, 255, 0), 2)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # 얼굴 위치 확인
            face_center_x = (face.left() + face.right()) // 2
            face_center_y = (face.top() + face.bottom()) // 2

            if not ((center_x - FACE_BOX_WIDTH // 2) < face_center_x < (center_x + FACE_BOX_WIDTH // 2) and
                    (center_y - FACE_BOX_HEIGHT // 2) < face_center_y < (center_y + FACE_BOX_HEIGHT // 2)):
                cv2.putText(frame, "Please align your face within the oval", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            leftEye = shape[lStart:lEnd + 1]
            rightEye = shape[rStart:rEnd + 1]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            mouth = shape[mStart:mEnd + 1]
            mar = calculate_mar(mouth)

            yaw, pitch, roll = get_euler_angles(shape, frame)

            if ear < EAR_THRESHOLD:
                eye_blink_counter += 1
            else:
                eye_blink_counter = 0

            if mar > YAWN_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            if abs(pitch) > PITCH_THRESHOLD:
                head_tilt_counter += 1
            else:
                head_tilt_counter = 0

            if eye_blink_counter >= FRAME_THRESHOLD or yawn_counter >= YAWN_FREQUENCY_THRESHOLD:
                if not alarm_active:
                    pygame.mixer.music.load(eye_closed_sound_path)
                    pygame.mixer.music.play()
                    alarm_active = True

            if alarm_active and (eye_blink_counter == 0 and yawn_counter == 0 and head_tilt_counter == 0):
                pygame.mixer.music.stop()
                alarm_active = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 스트리밍 처리 실행
process_stream()

pygame.mixer.quit()
