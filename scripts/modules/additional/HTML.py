import json
import os
from datetime import datetime

class HtmlManager:
    def __init__(self, 
                 template_path="template/template.html", 
                 output_path="FTP_files/index.html", 
                 json_path="template/tr_td_data.json"):
        self.template_path = template_path
        self.output_path = output_path
        self.json_path = json_path

    # **1. HTML 템플릿 로드**
    def load_template(self):
        """HTML 템플릿 로드 및 변환"""
        with open(self.template_path, "r", encoding="utf-8") as file:
            template = file.read()
            template = template.replace("{", "{{").replace("}", "}}")
            template = template.replace("{{Progress_Logs}}", "{Progress_Logs}")
            template = template.replace("{{Density_Data}}", "{Density_Data}")
        return template

    # **2. JSON 데이터 로드**
    def load_json_data(self):
        """JSON 데이터 로드"""
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {"Progress_Logs_html": "", "Density_Data_html": ""}

    # **3. JSON 데이터 저장**
    def save_json_data(self, progress_logs, density_data):
        """JSON 데이터 저장"""
        data = {
            "Progress_Logs_html": progress_logs,
            "Density_Data_html": density_data
        }
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    # **4. 로그 데이터 추가**
    def add_progress_log(self, progress_logs, process_name, step_name, status, details, max_entries=100):
        """진행 로그 데이터 추가"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        progress_logs += f"""
            <tr>
                <td>{current_time}</td>
                <td>{process_name}</td>
                <td>{step_name}</td>
                <td>{status}</td>
                <td>{details}</td>
            </tr>
        """
        progress_logs_rows = progress_logs.strip().split("\n<tr>")
        if len(progress_logs_rows) > max_entries:
            progress_logs_rows = progress_logs_rows[-max_entries:]
        return "\n<tr>".join(progress_logs_rows)

    # **5. 밀도 데이터 추가**
    def add_density_data(self, density_data, density, max_entries=100):
        """밀도 데이터 추가"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        density_data += f"""
            <tr>
                <td>{current_time}</td>
                <td>{density}</td>
            </tr>
        """
        density_data_rows = density_data.strip().split("\n<tr>")
        if len(density_data_rows) > max_entries:
            density_data_rows = density_data_rows[-max_entries:]
        return "\n<tr>".join(density_data_rows)

    # **6. HTML 저장**
    def save_html(self, template, progress_logs, density_data):
        """HTML 파일 생성 및 저장"""
        new_html = template.format(Progress_Logs=progress_logs, Density_Data=density_data)
        with open(self.output_path, "w", encoding="utf-8") as file:
            file.write(new_html)
        print(f"HTML updated and saved to {self.output_path}")

    # **7. 전체 실행 흐름**
    def append_html(self, process_name, step_name, status, details, density=None, max_entries=100):
        """전체 작업 실행"""
        # 템플릿 로드
        template = self.load_template()

        # JSON 데이터 로드
        html_data = self.load_json_data()
        progress_logs = html_data["Progress_Logs_html"]
        density_data = html_data["Density_Data_html"]

        # 데이터 추가
        progress_logs = self.add_progress_log(progress_logs, process_name, step_name, status, details, max_entries)
        if density is not None:
            density_data = self.add_density_data(density_data, density, max_entries)

        # JSON 저장
        self.save_json_data(progress_logs, density_data)

        # HTML 저장
        self.save_html(template, progress_logs, density_data)
