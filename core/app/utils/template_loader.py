def load_template(template_path):
    """HTML 템플릿 로드 및 변환"""
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
        template = template.replace("{", "{{").replace("}", "}}")
        template = template.replace("{{Progress_Logs}}", "{Progress_Logs}")
        template = template.replace("{{Density_Data}}", "{Density_Data}")
    return template