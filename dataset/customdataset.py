import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Optional, Tuple, Dict

class CustomDetectionDataset(Dataset):
    def __init__(self, data_set_path: str, transforms: Optional[transforms.Compose] = None) -> None:
        self.data_set_path = data_set_path
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(data_set_path) if f.endswith('.jpg')]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 이미지 파일 불러오기
        image_path = os.path.join(self.data_set_path, self.image_files[index])
        image = Image.open(image_path).convert("RGB")
        
        # 해당 이미지의 JSON 파일 불러오기
        annotation_file = os.path.splitext(image_path)[0] + '.json'
        with open(annotation_file) as f:
            annotation = json.load(f)
        
        # 바운딩 박스와 레이블 정보 가져오기
        bbox = annotation['bbox']
        label = annotation['category_id']
        
        # 바운딩 박스와 레이블을 텐서로 변환
        bbox = torch.tensor(bbox, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        # 변환 적용
        if self.transforms:
            image = self.transforms(image)
        
        target = {"boxes": bbox.unsqueeze(0), "labels": label.unsqueeze(0)}
        
        return image, target
