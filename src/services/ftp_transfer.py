import os
from ftplib import FTP

class FTPmanager:
    def __init__(self, ftp_server, ftp_user, ftp_password):
        # FTP 연결
        self.ftp = FTP(ftp_server)
        self.ftp.login(user=ftp_user, passwd=ftp_password)
        print("FTP 연결 성공!")

    def upload_files_to_ftp(self, directory_path, upload_dir):
        try:
            # 디렉토리에서 모든 파일 경로 가져오기
            file_paths = [
                os.path.join(directory_path, file_name)
                for file_name in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, file_name))
            ]

            if not file_paths:
                print("업로드할 파일이 없습니다.")
                return

            # 파일 전송
            for file_path in file_paths:
                try:
                    file_name = os.path.basename(file_path)  # 파일 이름 추출
                    upload_path = f"{upload_dir}/{file_name}"  # 업로드 경로 생성
                    with open(file_path, "rb") as file:
                        self.ftp.storbinary(f"STOR {upload_path}", file)
                    print(f"파일 업로드 성공: {file_name} -> {upload_path}")
                except Exception as file_error:
                    print(f"파일 업로드 중 오류 발생: {file_path}, 오류: {file_error}")

            # FTP 연결 종료
            self.ftp.quit()
            print("FTP 연결 종료.")
        except Exception as e:
            print(f"FTP 연결 중 오류 발생: {e}")
