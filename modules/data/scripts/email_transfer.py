import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class EmailManager:
    def __init__(self, sender_email, sender_password, recipient_email):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        
    def SendEmail(self, subject, body):
        # Gmail SMTP 서버 설정
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587

        # 이메일 메시지 설정
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            # SMTP 서버에 연결
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # TLS 암호화 시작
            server.login(self.sender_email, self.sender_password)  # 로그인
            server.send_message(msg)  # 이메일 보내기
            print("이메일이 성공적으로 발송되었습니다.")
        except Exception as e:
            print(f"이메일 발송 중 오류 발생: {e}")
        finally:
            server.quit()  # 서버 연결 종료

email_config={
    "sender_email": "kimjunhee2483@gmail.com", 
    "sender_password": "qtcv mtwy tagg bkvw" , 
    "recipient_email": "kimjunhee2483@gmail.com"
}
email_manager = EmailManager(**email_config)
email_manager.SendEmail("자동차통신시스템설계", 
                        "안녕하세요. 16주차 발표를 시작하겠습니다.")