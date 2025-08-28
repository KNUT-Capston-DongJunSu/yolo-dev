import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import convert_odgt_to_yolo

if __name__=="__main__":
    folder_path = './datasets/labels/'  # 예: 'C:/Users/you/data'

    json_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith('.json')
    ]

    for path in json_paths:
        if '/datasets/labels/208.Indoor_까페노아208(660)' in path:
            convert_odgt_to_yolo(
                odgt_path=path,
                images_path="./datasets/images/val",
                output_dir="./datasets/labels/val",
                class_ids={"bbox": 0}
            )
        else: 
            convert_odgt_to_yolo(
                odgt_path=path,
                images_path="./datasets/images/train",
                output_dir="./datasets/labels/train",
                class_ids={"bbox": 0}  
            )
