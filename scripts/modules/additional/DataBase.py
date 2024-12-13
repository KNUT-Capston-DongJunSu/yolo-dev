Simport mysql.connector
from datetime import datetime

class DatabaseManager:
    def __init__(self, host, user, password, database):
        """데이터베이스 연결 초기화"""
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()

    def insert_progresslogs(self, process_name, step_name, status, details):
        """진행 상황 로그를 삽입"""
        sql = """
        INSERT INTO ProgressLogs (timestamp, process_name, step_name, status, details)
        VALUES (%s, %s, %s, %s, %s)
        """
        timestamp = datetime.now()  # 현재 시간
        values = (timestamp, process_name, step_name, status, details)

        try:
            self.cursor.execute(sql, values)
            self.conn.commit()
            print(f"[ProgressLog] {process_name} - {step_name} - {status}")
        except Exception as e:
            print(f"[Error] Failed to insert progress log: {e}")

    def insert_errorlogs(self, error_type, message):
        """에러 로그를 삽입"""
        sql = """
        INSERT INTO ErrorLogs (timestamp, error_type, message)
        VALUES (%s, %s, %s)
        """
        timestamp = datetime.now()  # 현재 시간
        values = (timestamp, error_type, message)

        try:
            self.cursor.execute(sql, values)
            self.conn.commit()
            print(f"[ErrorLog] {error_type} - {message}")
        except Exception as e:
            print(f"[Error] Failed to insert error log: {e}")

    def close(self):
        """데이터베이스 연결 종료"""
        self.cursor.close()
        self.conn.close()
