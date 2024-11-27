
def calculate_total_density(near_density, mid_density, far_density, weights):
    """
    near_density, mid_density, far_density: 각 구역별 밀도
    weights: 각 구역의 비중 (ex: [0.5, 0.3, 0.2])
    반환값: 전체 밀도 값
    """
    total_density = (
        near_density * weights[0] +
        mid_density * weights[1] +
        far_density * weights[2]
    )
    return total_density

def calculate_scale_factor(camera_height, fov, base_area=100):
    """
    camera_height: 카메라 높이
    fov: 카메라 시야각 정보
    base_area: 기준 면적 (100)
    반환값: Scale Factor
    """
    scale_factor = (camera_height * fov) / base_area
    return scale_factor

def calculate_density(objects, region_area):
    """
    objects: 해당 구역 내 감지된 객체 리스트
    region_area: 구역 면적
    반환값: 밀도 값
    """
    object_count = len(objects)  # 객체 수
    density = object_count / region_area  # 밀도 계산
    return density

def assign_region(bounding_boxes, thresholds):
    """
    bounding_boxes: [[x1, y1, x2, y2, conf, class], ...] 형태의 바운딩 박스 리스트
    thresholds: [near, mid] 형태로 구역 기준 거리
    반환값: 각 구역별 객체 리스트
    """
    near_objects = []
    mid_objects = []
    far_objects = []
    
    for box in bounding_boxes:
        height = box[3] - box[1]  # 바운딩 박스 높이 계산
        if height >= thresholds[0]:
            near_objects.append(box)
        elif thresholds[0] > height >= thresholds[1]:
            mid_objects.append(box)
        else:
            far_objects.append(box)
    
    return near_objects, mid_objects, far_objects

# 바운딩 박스 (예제 데이터)
bounding_boxes = [[10, 20, 50, 80, 0.9, 'person'], [30, 40, 70, 90, 0.8, 'person']]
thresholds = [50, 30]  # 가까운/중간 기준 바운딩 박스 높이 임계값
region_areas = [200, 300, 400]  # 구역별 면적

# 카메라 설정
camera_height = 3  # 카메라 높이 (m)
fov = 60  # 시야각 (degrees)

# 단계별 실행
near_objects, mid_objects, far_objects = assign_region(bounding_boxes, thresholds)
near_density = calculate_density(near_objects, region_areas[0])
mid_density = calculate_density(mid_objects, region_areas[1])
far_density = calculate_density(far_objects, region_areas[2])

scale_factor = calculate_scale_factor(camera_height, fov)
weights = [0.5, 0.3, 0.2]  # 가중치
total_density = calculate_total_density(near_density, mid_density, far_density, weights)

print("구역별 밀도:", near_density, mid_density, far_density)
print("전체 밀도:", total_density)
