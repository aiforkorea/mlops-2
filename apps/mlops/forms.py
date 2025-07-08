# apps/mlops/forms.py
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    # 파일 업로드 필드 (required=True로 설정하면 필수 필드가 됨)
    file = FileField('파일 선택', validators=[DataRequired()])
    submit = SubmitField('업로드')