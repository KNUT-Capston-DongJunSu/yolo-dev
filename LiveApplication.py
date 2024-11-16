import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from main import SimpleObjectDetector
from typing import List

class ApplicationHandler:
    def __init__(self) -> None:
        # 모델 인스턴스화
        num_classes = 3  # 예: 배경 클래스 + 2개의 객체 클래스
        self.model = SimpleObjectDetector(num_classes=num_classes)
        self.model.load_state_dict(torch.load('trained_detector.pth'))
        self.model.eval()  # 모델을 평가 모드로 설정

        # PiCamera 초기화
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))

        # 이미지 전처리 함수
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 모델 입력 크기에 맞추기
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 이미지넷 정규화 값 사용
        ])

    def Run(self) -> None:
        # 배경 차분기로 초기 바운딩 박스 설정
        self.backSub = cv2.createBackgroundSubtractorMOG2()

        # 카메라가 초기화되기까지 잠시 대기
        time.sleep(0.1)

        # 프레임 스트림 시작
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            original_image = image.copy()  # 디스플레이용 원본 이미지 저장

            # 배경 차분을 통해 객체 탐지
            fg_mask = self.backSub.apply(image)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 가장 큰 객체의 바운딩 박스를 얻어 proposals로 설정
            proposals: List[List[int]] = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                proposals.append([x, y, x + w, y + h])

            # proposals가 비어있는 경우 continue
            if not proposals:
                print("객체를 찾지 못했습니다.")
                self.rawCapture.truncate(0)
                continue

            # 이미지 전처리
            input_image = self.transform(image)
            input_image = input_image.unsqueeze(0)  # 배치 차원 추가

            # proposals를 모델 입력 크기에 맞게 조정
            scale_x = 224 / image.shape[1]
            scale_y = 224 / image.shape[0]
            scaled_proposals = [
                [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
                for (x1, y1, x2, y2) in proposals
            ]
            scaled_proposals = torch.tensor(scaled_proposals).unsqueeze(0)  # 배치 차원 추가

            # 모델 실행
            with torch.no_grad():
                cls_scores, bbox_preds, objectness, bbox_deltas = self.model(input_image, scaled_proposals)

            print("Class Scores:", cls_scores)
            print("Bounding Box Predictions:", bbox_preds)

            # 바운딩 박스를 프레임에 그리기 (시각화)
            for proposal in proposals:  # 원본 크기의 proposals 사용
                x1, y1, x2, y2 = map(int, proposal)
                cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 결과 디스플레이
            cv2.imshow("Tracking", original_image)

            # 프레임 초기화
            self.rawCapture.truncate(0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main = ApplicationHandler()
    main.Run()
