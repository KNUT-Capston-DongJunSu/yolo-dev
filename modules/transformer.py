import os
import json
from PIL import Image

def convert_odgt_to_yolo(odgt_path, images_path, output_dir, class_ids={"fbox": 0, "hbox": 1}):
    os.makedirs(output_dir, exist_ok=True)
    converted_files = 0
    ignored_boxes = 0
    mismatched_images = []

    with open(odgt_path, 'r', encoding='utf-8-sig') as f:
        entry = json.load(f)
        for line in entry['image']:
            try:
                image_id: str = line['imagename']
                gtboxes = line.get("crowdinfo", []).get("objects", [])
                image_path = os.path.join(images_path, f"{image_id}")

                # 이미지 확인
                if not os.path.exists(image_path):
                    print(f"[Warning] 이미지 파일 누락: {image_path}")
                    mismatched_images.append(image_id)
                    continue

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                    # YOLO 레이블 파일 생성
                    image_name = image_id.replace('.jpg', '')
                    label_file_path = os.path.join(output_dir, f"{image_name}.txt")
                    processed_boxes = set()  # 중복 제거를 위한 기록 저장소

                    with open(label_file_path, 'w') as label_file:
                        for box in gtboxes:
                            # 중복 확인 및 처리
                            bbox = tuple(box.get("bbox", []))
                            # fbox 처리 (클래스 ID 0)
                            if len(bbox) == 4 and bbox not in processed_boxes:
                                processed_boxes.add(bbox)  # 중복 방지
                                x1, y1, width, height = bbox
                                x_center = (x1 + width / 2) / img_width
                                y_center = (y1 + height / 2) / img_height
                                width /= img_width
                                height /= img_height
                                x_center = max(0, min(1, x_center))
                                y_center = max(0, min(1, y_center))
                                width = max(0, min(1, width))
                                height = max(0, min(1, height))
                                label_file.write(f"{class_ids['bbox']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                converted_files += 1

            except Exception as e:
                print(f"[Error] 변환 중 오류 발생: {line}\n{e}")

    # 변환 요약 출력
    print(f"[Info] 변환 완료: {converted_files}개 파일")
    print(f"[Info] 무시된 박스: {ignored_boxes}개")
    if mismatched_images:
        print(f"[Warning] 누락된 이미지: {len(mismatched_images)}개")


import os
import xml.etree.ElementTree as ET



def voc_xmls_to_yolo_txts(xml_dir, output_dir, class_map={"person": 0}):
    os.makedirs(output_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    total_files = len(xml_files)
    converted = 0

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            filename = root.find('filename').text
            num_str = os.path.splitext(filename)[0]  # "11"
            num = int(num_str)  # 숫자 변환
            txt_filename = 'PartA_' + f'{num:05d}' + '.txt'
            txt_path = os.path.join(output_dir, txt_filename)

            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    cls_name = obj.find('name').text
                    if cls_name not in class_map:
                        continue

                    cls_id = class_map[cls_name]

                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    x_center = ((xmin + xmax) / 2) / width
                    y_center = ((ymin + ymax) / 2) / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height

                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            converted += 1

        except Exception as e:
            print(f"[Error] {xml_file} 변환 실패: {e}")

    print(f"[완료] 총 {total_files}개 중 {converted}개 변환 성공")

# 사용 예시

    
    
