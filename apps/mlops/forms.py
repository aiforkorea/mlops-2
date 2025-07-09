# apps/mlops/forms.py

from flask_wtf import FlaskForm
from wtforms import FileField, SelectField, SelectMultipleField, StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import ListWidget, CheckboxInput

class UploadForm(FlaskForm):
    file = FileField('파일 선택', validators=[DataRequired()])
    submit = SubmitField('업로드')

class ImputeForm(FlaskForm):
    impute_method = SelectField(
        '결측치 보정 방법',
        choices=[],
        render_kw={"class": "form-control"}
    )
    submit = SubmitField('적용')

# 체크박스 필드용 커스텀 위젯 정의
class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()

class FeatureTargetForm(FlaskForm):
    features = MultiCheckboxField('피처(Feature)', choices=[], validators=[DataRequired()])
    target = SelectField('타겟(Target)', choices=[], validators=[DataRequired()])
    submit = SubmitField('다음 (모델학습)')

class AddModelForm(FlaskForm):
    name = StringField('이름', validators=[DataRequired()])
    model_type = SelectField('종류', choices=[], validators=[DataRequired()])
    submit = SubmitField('등록')