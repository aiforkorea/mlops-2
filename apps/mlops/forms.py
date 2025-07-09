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

class FeatureSelectionForm(FlaskForm):
    features = SelectMultipleField(
        '피처(Feature)',
        widget=ListWidget(prefix_label=False),
        option_widget=CheckboxInput()
    )
    target = SelectField('타겟(Target)')
    submit = SubmitField('다음 (모델학습)')

class AddModelForm(FlaskForm):
    """Form for adding a new machine learning model."""
    name = StringField(
        '이름', 
        validators=[DataRequired(message="모델 이름은 필수입니다.")]
    )
    model_type = SelectField('종류', validators=[DataRequired()])
    submit = SubmitField('등록')

class TrainModelForm(FlaskForm):
    """Form for selecting a model to train."""
    model_id = SelectField(
        '사용 모델',
        validators=[DataRequired()],
        coerce=int
    )
    submit = SubmitField('교차검증 및 학습')


    