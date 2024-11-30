import math
import time  # 실시간 시뮬레이션용

def calculate_camera_distance(camera_height, y_bottom, frame_height, fov_vertical):
    # 객체의 각도 계산
    theta = math.radians(fov_vertical / 2) * (1 - y_bottom / frame_height)
    
    # 거리 계산
    distance = camera_height / math.tan(theta)
    return distance

def calculate_real_height(object_pixel_height, frame_height, fov_vertical, camera_distance):
    # 카메라의 투영 영역 높이 계산
    h_view = 2 * camera_distance * math.tan(math.radians(fov_vertical / 2))
    
    # 실제 높이 계산
    real_height = (object_pixel_height / frame_height) * h_view
    return real_height

def calculate_region_volume(camera_height, fov_horizontal, fov_vertical, max_height):
    # 너비 계산
    width = 2 * camera_height * math.tan(math.radians(fov_horizontal / 2))
    
    # 높이 계산
    area_height = 2 * camera_height * math.tan(math.radians(fov_vertical / 2))
    
    # 부피 계산
    volume = width * area_height * max_height
    return volume, width, area_height

# ===== 실시간 시뮬레이션 =====

# Pi Camera 기본 시야각
fov_horizontal = 62.2  # degrees
fov_vertical = 48.8  # degrees

# 카메라 높이 및 해상도
camera_height = 3.0  # 카메라 높이 (m)
frame_height = 1080  # 세로 픽셀

# 프레임별 객체 바운딩 박스 정보 (실시간 시뮬레이션용)
frames = [
    [{"y_bottom": 800, "pixel_height": 150}, {"y_bottom": 700, "pixel_height": 300}],
    [{"y_bottom": 850, "pixel_height": 200}, {"y_bottom": 900, "pixel_height": 250}],
    [{"y_bottom": 750, "pixel_height": 400}, {"y_bottom": 800, "pixel_height": 150}],
]

# 초기값 설정
alpha = 0.8  # 지수 이동 평균 가중치

for frame_idx, bounding_boxes in enumerate(frames):
    print(f"\n=== 프레임 {frame_idx + 1} ===")

    # 객체별 거리와 실제 높이 계산
    object_heights = []
    for obj in bounding_boxes:
        y_bottom = obj["y_bottom"]
        pixel_height = obj["pixel_height"]
        
        # 카메라와 객체 간 거리 계산
        camera_distance = calculate_camera_distance(
            camera_height, y_bottom, frame_height, fov_vertical)
        
        # 객체 실제 높이 계산
        real_height = calculate_real_height(
            pixel_height, frame_height, fov_vertical, camera_distance)
        object_heights.append(real_height)

    # 현재 프레임에서 가장 높은 객체 찾기
    current_max_height = max(object_heights)
    print(f"현재 프레임 최대 높이: {current_max_height:.2f} m")

    if frame_idx == 0:
        previous_max_height = current_max_height
        
    # 스무딩 적용
    max_height = alpha * previous_max_height + (1 - alpha) * current_max_height
    previous_max_height = max_height
    print(f"스무딩 후 최대 높이: {max_height:.2f} m")

    # 관찰 구역 부피 계산
    volume, width, area_height = calculate_region_volume(
        camera_height, fov_horizontal, fov_vertical, max_height)
    print(f"구역 부피: {volume:.2f} ㎥")
    print(f"구역 너비: {width:.2f} m")
    print(f"구역 높이 (면적용): {area_height:.2f} m")

    # 객체 수 및 혼잡도 계산
    object_count = len(bounding_boxes)
    density = object_count / volume
    print(f"객체 수: {object_count}, 혼잡도: {density:.2f} 객체/㎥")

    # 시뮬레이션 속도 조절 (실제 환경에서는 제거)
    time.sleep(1)
