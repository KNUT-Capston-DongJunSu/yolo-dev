import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Optional, Tuple, Dict

class CustomDetectionDataset(Dataset):
    def __init__(self, data_set_path: str, odgt_file: str, transforms: Optional[transforms.Compose] = None) -> None:
        self.data_set_path = data_set_path
        self.transforms = transforms
        self.image_files = []
        self.annotations = []

        # odgt 파일 읽기
        with open(odgt_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                self.image_files.append(f"{entry['ID']}.jpg")
                self.annotations.append(entry['gtboxes'])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 이미지 파일 불러오기
        image_path = os.path.join(self.data_set_path, self.image_files[index])
        image = Image.open(image_path).convert("RGB")
        
        # 해당 이미지의 어노테이션 가져오기
        annotation = self.annotations[index]
        
        # 여러 바운딩 박스와 레이블 정보 가져오기
        bboxes = []
        labels = []
        objectness = []  # 객체 존재 여부를 위한 리스트 추가
        for box in annotation:
            bboxes.append(box['hbox'])  # 바운딩 박스 좌표
            labels.append(box['tag'])  # 클래스 레이블
            
            # 객체가 존재하는 경우 (예: 'person') 1, 객체가 없는 경우 0
            # tag가 'person'인 경우만 객체로 가정하고 objectness를 1로 설정
            if box['tag'] == 'person':
                objectness.append(1)  # 객체 존재
            else:
                objectness.append(0)  # 객체가 없음

        # 클래스 레이블 매핑 (예: "person" -> 1, "background" -> 0)
        label_map = {'person': 1, 'background': 0}  # 배경을 0으로 설정
        labels = [label_map.get(label, 0) for label in labels]  # 기본값: 0 (알 수 없는 레이블)

        # 바운딩 박스와 레이블을 텐서로 변환
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        objectness = torch.tensor(objectness, dtype=torch.float32)  # objectness 값을 텐서로 변환

        # 변환 적용
        if self.transforms:
            image = self.transforms(image)

        # 타겟 딕셔너리 반환
        target = {"boxes": bboxes, "labels": labels, "objectness": objectness}

        return image, target

