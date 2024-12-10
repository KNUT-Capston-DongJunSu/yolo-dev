import os
import json
from PIL import Image

def convert_odgt_to_yolo(odgt_path, images_path, output_dir, class_ids={"fbox": 0, "hbox": 1}):
    os.makedirs(output_dir, exist_ok=True)
    converted_files = 0
    ignored_boxes = 0
    mismatched_images = []

    with open(odgt_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                image_id = entry["ID"]
                gtboxes = entry.get("gtboxes", [])
                image_path = os.path.join(images_path, f"{image_id}.jpg")

                # 이미지 확인
                if not os.path.exists(image_path):
                    print(f"[Warning] 이미지 파일 누락: {image_path}")
                    mismatched_images.append(image_id)
                    continue

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                    # YOLO 레이블 파일 생성
                    label_file_path = os.path.join(output_dir, f"{image_id}.txt")
                    processed_boxes = set()  # 중복 제거를 위한 기록 저장소

                    with open(label_file_path, 'w') as label_file:
                        for box in gtboxes:
                            if box.get("tag") != "person":
                                continue  # 'person' 클래스만 처리
                            if box.get("extra", {}).get("ignore", 0) == 1:
                                ignored_boxes += 1
                                continue

                            # 중복 확인 및 처리
                            fbox = tuple(box.get("fbox", []))
                            hbox = tuple(box.get("hbox", []))

                            # fbox 처리 (클래스 ID 0)
                            if len(fbox) == 4 and fbox not in processed_boxes:
                                processed_boxes.add(fbox)  # 중복 방지
                                x1, y1, width, height = fbox
                                x_center = (x1 + width / 2) / img_width
                                y_center = (y1 + height / 2) / img_height
                                width /= img_width
                                height /= img_height
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                width = max(0, min(1, width))
                                height = max(0, min(1, height))
                                label_file.write(f"{class_ids['fbox']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                            # hbox 처리 (클래스 ID 1)
                            if len(hbox) == 4 and hbox not in processed_boxes:
                                processed_boxes.add(hbox)  # 중복 방지
                                x1, y1, width, height = hbox
                                x_center = (x1 + width / 2) / img_width
                                y_center = (y1 + height / 2) / img_height
                                width /= img_width
                                height /= img_height
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                width = max(0, min(1, width))
                                height = max(0, min(1, height))
                                label_file.write(f"{class_ids['hbox']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                converted_files += 1

            except Exception as e:
                print(f"[Error] 변환 중 오류 발생: {line}\n{e}")

    # 변환 요약 출력
    print(f"[Info] 변환 완료: {converted_files}개 파일")
    print(f"[Info] 무시된 박스: {ignored_boxes}개")
    if mismatched_images:
        print(f"[Warning] 누락된 이미지: {len(mismatched_images)}개")

# 훈련 데이터 변환
convert_odgt_to_yolo(
    odgt_path="./datasets/annotation_train.odgt",
    images_path="./datasets/Images/train",
    output_dir="./datasets/labels/train",
    class_ids={"fbox": 0, "hbox": 1}  # 두 클래스: fbox=0, hbox=1
)

# 검증 데이터 변환
convert_odgt_to_yolo(
    odgt_path="./datasets/annotation_val.odgt",
    images_path="./datasets/Images/val",
    output_dir="./datasets/labels/val",
    class_ids={"fbox": 0, "hbox": 1}  # 두 클래스: fbox=0, hbox=1
)
