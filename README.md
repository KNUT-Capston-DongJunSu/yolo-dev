# DensityAI
간략히 이 시스템은 [CrowdHuman](https://www.crowdhuman.org/)의 datasets을 다운로드 받아 
[ultralytics](https://github.com/ultralytics)의 YOLOv5 모델을 사용하며
프레임당 영상을 분석하여 특정 지역의 혼잡도를 실시간으로 탐지하고 알림을 제공하는 AI 기반 객체 탐지 시스템입니다. 

## 주요 기능
- 이미지 또는 실시간 영상에서 사람의 밀도를 추정
- 특정 임계값 이상일 경우 알림 제공
- 모델 학습을 위한 데이터셋 구성 및 전처리 기능 포함

## 설치 방법
이 저장소를 클론합니다.
git clone https://github.com/username/DensityAI.git
cd CongestionNotifier
필요한 패키지를 설치합니다. requirements.txt 파일을 통해 필요한 라이브러리를 자동으로 설치할 수 있습니다.
pip install -r requirements.txt

## 실시간 애플리케이션 실행
main.py 스크립트를 실행하여 실시간으로 혼잡도를 모니터링하고 알림을 받을 수 있습니다.
python main.py



