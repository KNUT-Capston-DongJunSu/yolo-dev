import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from densEstAI.utils import voc_xmls_to_yolo_txts

voc_xmls_to_yolo_txts('./SCUT-HEAD/labels/val/', './SCUT-HEAD/labels/val/')
