import os
import json
from datetime import datetime
from densEstAI.core.utils.template_loader import load_template
from densEstAI.core.utils.json_handler import load_json_data
from densEstAI.core.utils.json_handler import save_json_data
from densEstAI.core.utils.html_handler import save_html

class HtmlGenerator:
    def __init__(self, template_path="template/template.html", output_path="FTP_files/index.html", json_path="template/tr_td_data.json"):
        self.template_path = template_path
        self.output_path = output_path
        self.json_path = json_path

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

    # **7. 전체 실행 흐름**
    def append_html(self, process_name, step_name, status, details, density=None, max_entries=100):
        """전체 작업 실행"""
        # 템플릿 로드
        template = load_template(self.template_path)

        # JSON 데이터 로드
        html_data = load_json_data(self.json_path)
        progress_logs = html_data["Progress_Logs_html"]
        density_data = html_data["Density_Data_html"]

        # 데이터 추가
        progress_logs = self.add_progress_log(progress_logs, process_name, step_name, status, details, max_entries)
        if density is not None:
            density_data = self.add_density_data(density_data, density, max_entries)

        # JSON 저장
        save_json_data(progress_logs, density_data)

        # HTML 저장
        save_html(self.output_path, template, progress_logs, density_data)
