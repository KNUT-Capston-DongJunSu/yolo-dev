import math

class ObjectDimensionEstimator:
    def __init__(self, camera_height, frame_height, fov_horizontal, fov_vertical):
        self.fov_horizontal = fov_horizontal  # degrees
        self.fov_vertical = fov_vertical  # degrees

        # 카메라 높이 및 해상도
        self.camera_height = camera_height
        self.frame_height = frame_height
        
        self.alpha = 0.8  # 지수 이동 평균 가중치
        self.previous_max_height = None

    def _calculate_camera_distance(self, y_bottom):
        # 객체의 각도 계산
        theta = math.radians(self.fov_vertical / 2) * (1 - y_bottom / self.frame_height)
        
        # 거리 계산
        distance = self.camera_height / math.tan(theta)
        return distance

    def _calculate_real_height(self, object_pixel_height, camera_distance):
        # 카메라의 투영 영역 높이 계산
        h_view = 2 * camera_distance * math.tan(math.radians(self.fov_vertical / 2))
        
        # 실제 높이 계산
        real_height = (object_pixel_height / self.frame_height) * h_view
        return real_height

    def _calculate_region_volume(self, max_height):
        # 너비 계산
        width = 2 * self.camera_height * math.tan(math.radians(self.fov_horizontal / 2))
        
        # 높이 계산
        area_height = 2 * self.camera_height * math.tan(math.radians(self.fov_vertical / 2))
        
        # 부피 계산
        volume = width * area_height * max_height

        return volume, width, area_height

    def _extract_object_dimensions(self, predictions):
        results = []
        for box in predictions:
            y_min, y_max = box[1], box[3]  # y_min, y_max 추출
            y_bottom = y_max
            pixel_height = y_max - y_min
            results.append({"y_bottom": y_bottom, "pixel_height": pixel_height})
        return results

    def _calculate_maximum_height(self, bounding_boxes):
        if len(bounding_boxes)==0:  # 객체가 탐지되지 않은 경우
            print("No objects detected in the current frame.")
            return 0  # 밀도를 0으로 반환

        # 객체별 거리와 실제 높이 계산
        object_heights = []
        for obj in bounding_boxes:
            y_bottom = obj["y_bottom"]
            pixel_height = obj["pixel_height"]
            
            # 카메라와 객체 간 거리 계산
            camera_distance = self._calculate_camera_distance(y_bottom)
            # 객체 실제 높이 계산
            real_height = self._calculate_real_height(pixel_height, camera_distance)
            object_heights.append(real_height)

        # 현재 프레임에서 가장 높은 객체 찾기
        current_max_height = max(object_heights)
        return current_max_height

    def _smooth_max_height(self, current_max_height):
        if self.previous_max_height is None:
            self.previous_max_height = current_max_height
            
        # 스무딩 적용
        max_height = (1 - self.alpha) * self.previous_max_height + self.alpha * current_max_height
        self.previous_max_height = max_height

        return max_height

class DensityEstimator(ObjectDimensionEstimator):
    def __init__(self, camera_height, frame_height, fov_horizontal=65, fov_vertical=49):
        super().__init__(camera_height, frame_height, fov_horizontal, fov_vertical)

    def calculate_density(self, predictions):
        # 프레임별 객체 바운딩 박스 정보 (실시간 시뮬레이션용)
        bounding_boxes = self._extract_object_dimensions(predictions)
        current_max_height = self._calculate_maximum_height(bounding_boxes)        
        max_height = self._smooth_max_height(current_max_height)
        
        # 관찰 구역 부피 계산
        volume, width, area_height = self._calculate_region_volume(max_height)
        
        # 객체 수 및 혼잡도 계산
        object_count = len(bounding_boxes)
        density = object_count / volume
        
        print(f"현재 프레임 최대 높이: {current_max_height:.2f} m")
        print(f"스무딩 후 최대 높이: {max_height:.2f} m")
        print(f"구역 부피: {volume:.2f} ㎥")
        print(f"구역 너비: {width:.2f} m")
        print(f"구역 높이 (면적용): {area_height:.2f} m")
        print(f"객체 수: {object_count}, 혼잡도: {density:.2f} 객체/㎥")

        return density