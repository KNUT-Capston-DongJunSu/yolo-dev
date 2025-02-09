from modules.app import ApplicationHandler
from modules.app import AdditionalApplicationHandler

if __name__ == '__main__':
    app = ApplicationHandler("results/train_preprocessing/weights/best.pt")
    app.app_start_running()

    db_config = {
        "host": "localhost",
        "user": "your_user",
        "password": "your_password",
        "database": "your_database"
    }
    email_config = {
        "sender_email": "your_email@gmail.com",
        "sender_password": "your_email_password",
        "recipient_email": "recipient_email@gmail.com"
    }
    ftp_config = {
        "ftp_server": "ftp_server",
        "ftp_user": "user_nmae",
        "ftp_password": "password"
    }
    app_add = AdditionalApplicationHandler(
        "results/train_preprocessing/weights/best.pt",
        db_config, email_config, ftp_config
    )
    app_add.app_start_running()