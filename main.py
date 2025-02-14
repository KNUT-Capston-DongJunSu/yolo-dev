import argparse
from modules.app import ApplicationHandler
from modules.app import AdditionalApplicationHandler
from configs.config import DB_CONFIG, FTP_CONFIG, EMAIL_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Application Handler Runner")

    parser.add_argument(
        "--model-path", type=str, default="results/train_preprocessing/weights/best.pt",
        help="Path to the model weights file"
    )

    # --add 옵션을 주었을 때만 AdditionalApplicationHandler 실행
    parser.add_argument(
        "--add", action="store_true",
        help="Enable AdditionalApplicationHandler"
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.add:
        # --add 옵션이 있을 때 실행
        app_add = AdditionalApplicationHandler(
            args.model_path, DB_CONFIG, EMAIL_CONFIG, FTP_CONFIG
        )
        app_add.app_start_running()
    else:
        # --add 옵션이 없을 때 실행
        app = ApplicationHandler(args.model_path)
        app.app_start_running()
