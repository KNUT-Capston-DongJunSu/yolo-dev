# YOLO-Density-Notifying
## 서비스 이름: DensityAI

## 개요
간략히 이 시스템은 [CrowdHuman](https://www.crowdhuman.org/)의 datasets을 다운로드 받아 
[ultralytics](https://github.com/ultralytics)의 YOLOv5 모델을 사용하며
프레임당 영상을 분석하여 특정 지역의 혼잡도를 실시간으로 탐지하고 알림을 제공하는 AI 기반 객체 탐지 시스템입니다. 

## 폴더 구조

```
├── configs/           # 설정 관련 파일 저장
├── modules/           # 기능별 모듈 저장
├── results/           # 분석 결과 저장
├── src/
│   ├── app/           # application 핵심 파일
│   ├── services/      # 웹 서비스와 이메일 알리미 구성 
│   ├── MyTransform.py # yolo포멧 변경
│   ├── shared_sort.py # 객체 트래킹
│   ├── yolo_trainer.py # yolo 모델 import하여 학습과 추론
├── template/          # 템플릿 파일 저장
├── .gitignore         # Git에서 제외할 파일 목록
├── main.py            # 메인 실행 파일
├── README.md          # 프로젝트 개요 및 설명 파일
```

## 실행 방법

1. YOLO 모델 학습 실행
```bash
python yolo_trainer.py
```

2. 메인 프로그램 실행
```bash
python main.py
```

## 데이터 변환
`odgt2yolo.py`를 사용하여 ODGT 형식의 데이터를 YOLO 학습 데이터로 변환할 수 있습니다.
```bash
python src/Mytransform.py
```