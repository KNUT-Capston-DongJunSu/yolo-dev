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
                self.image_files.append(entry['ID'])
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
        for box in annotation['gtboxes']:
            bboxes.append(box['hbox'])  # 바운딩 박스 좌표
            labels.append(box['tag'])  # 클래스 레이블
        
        # 클래스 레이블 매핑 (예: "person" -> 1)
        label_map = {'person': 1}  # 레이블 매핑 정의
        labels = [label_map.get(label, 0) for label in labels]  # 기본값: 0 (알 수 없는 레이블)

        # 바운딩 박스와 레이블을 텐서로 변환
        bboxs = torch.tensor(bboxs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 변환 적용
        if self.transforms:
            image = self.transforms(image)
        
        target = {"boxes": bboxs.unsqueeze(0), "labels": labels.unsqueeze(0)}
        
        return image, target
