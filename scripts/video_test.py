import os
import sys
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.core import ThreadedVideoStreamer
 
if __name__=='__main__':
    streamer = ThreadedVideoStreamer(
        video_path="./datasets/test/video/test11.mp4", 
        #v model_path="results/medium.pt",
        model_path="scut_head.pt",
        output_name="predict.mp4",
        camera_height=2.7
        )
    
    try: 
        streamer.start_stream()
        
    except Exception as e:
        print(f"[ERROR]: {e}")
    
    finally:
        streamer.stop_stream()

'''
우리 모델 
 [1] 객체 수 통계
  - 평균 객체 수: 11.53
  - 표준편차: 1.92

[2] 프레임 간 변화율
  - 평균 변화량 (ΔN): 0.35
  - 변화량 표준편차: 0.56

[3] 히트맵 기반 밀도
  - 히트맵 총합 평균: 2047.72
  - 히트맵 평균 강도: 0.50
'''

'''
스캇 모델
[1] 객체 수 통계
  - 평균 객체 수: 13.99
  - 표준편차: 1.83

[2] 프레임 간 변화율
  - 평균 변화량 (ΔN): 0.61
  - 변화량 표준편차: 0.72

[3] 히트맵 기반 밀도
  - 히트맵 총합 평균: 2048.08
  - 히트맵 평균 강도: 0.50
'''

# 스캇 모델은 객체를 더 많이 예측함 → 다소 민감한 탐지 경향
# 평균보다 더 많이 탐지되지만, 변화폭은 우리 모델보다 약간 더 작음 → 균일성은 유지

# 우리 모델이 프레임 간 예측 변화가 적고 안정적 → 영상 흐름 내 일관성 우수
# 스캇 모델은 변화량이 크고 들쭉날쭉 → 불필요한 감지나 누락 가능성 있음

# 둘 다 히트맵 상에서 비슷한 수준의 밀도와 강도를 갖고 있음
# 예측 위치 자체는 비슷하나, 객체 수 예측에서 차이 발생


'''
우리 모델	- 탐지 수는 약간 적지만 더 안정적이고, 변화량도 작음
- 프레임 간 일관성 우수
스캇 모델	- 더 많은 객체를 탐지하나, 프레임 간 예측 변동성이 큼
- 예측이 민감하고 불안정 가능성

정확도 중심이라면 → 우리 모델
민감도 중심(검출 누락 방지)**라면 → 스캇 모델
'''