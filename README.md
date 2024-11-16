# CongestionNotifier
혼잡도 알리미(Congestion Notifier)는 카메라 피드를 분석하여 특정 지역의 혼잡도를 실시간으로 탐지하고 알림을 제공하는 AI 기반 객체 탐지 시스템입니다. 이 시스템은 딥러닝 모델을 사용하여 사람이나 차량의 밀집도를 추정하고, 혼잡한 상황을 감지합니다.

## 주요 기능
- 이미지 또는 실시간 영상에서 객체(예: 사람, 차량 등)의 밀도를 추정
- 특정 임계값 이상일 경우 알림 제공
- 모델 학습을 위한 데이터셋 구성 및 전처리 기능 포함

## 폴더 구조
```plaintext
CongestionNotifier/
├── data/                    # 데이터 관련 폴더
│   ├── images/              # 원본 이미지 또는 테스트 이미지 폴더
│   ├── annotations/         # JSON 어노테이션 파일 저장
│   └── processed/           # 전처리된 데이터 저장
├── src/                     # 소스 코드 폴더
│   ├── customdataset.py     # 데이터셋 처리 클래스 파일
│   ├── model.py             # 모델 정의 파일
│   ├── train.py             # 모델 훈련 코드 파일
│   ├── test.py              # 테스트 코드 파일
│   └── LiveApplication.py  # 실시간 애플리케이션 실행 코드 파일
├── checkpoints/             # 훈련된 모델 가중치 저장
│   └── trained_detector.pth # 훈련된 모델 파일
├── requirements.txt         # 필요한 패키지 목록
└── README.md                # 프로젝트 설명 파일
```

## 설치 방법
이 저장소를 클론합니다.
git clone https://github.com/username/CongestionNotifier.git
cd CongestionNotifier
필요한 패키지를 설치합니다. requirements.txt 파일을 통해 필요한 라이브러리를 자동으로 설치할 수 있습니다.
pip install -r requirements.txt

## 사용 방법
1. 모델 학습
train.py 스크립트를 실행하여 모델을 학습시킵니다. 모델 학습이 완료되면 가중치 파일이 checkpoints/ 폴더에 저장됩니다.
python src/train.py

2. 모델 테스트
test.py 스크립트를 실행하여 모델 성능을 테스트합니다. 모델이 예측한 바운딩 박스와 정답을 비교하여 IoU 등의 지표를 계산합니다.
python src/test.py

3. 실시간 애플리케이션 실행
LiveApplication.py 스크립트를 실행하여 실시간으로 혼잡도를 모니터링하고 알림을 받을 수 있습니다.
python src/LiveApplication.py

## 요구사항
Python 3.8 이상
PyTorch, torchvision, OpenCV 등 (자세한 요구사항은 requirements.txt 참조)


