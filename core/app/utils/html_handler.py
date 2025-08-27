def save_html(output_path, template, progress_logs, density_data):
    """HTML 파일 생성 및 저장"""
    new_html = template.format(Progress_Logs=progress_logs, Density_Data=density_data)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(new_html)
    print(f"HTML updated and saved to {output_path}")