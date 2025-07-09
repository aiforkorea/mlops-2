# apps/mlops/forms.py

from flask_wtf import FlaskForm
from wtforms import FileField, FloatField, IntegerField, SelectField, SelectMultipleField, StringField, SubmitField
from wtforms.validators import DataRequired, NumberRange
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

def create_inference_form(input_cols):
    """
    입력 컬럼(피처)에 따라 동적으로 모델 추론용 FlaskForm을 생성합니다.
    """
    # 폼 필드를 저장할 딕셔너리를 먼저 생성합니다.
    # WTForms 필드 객체들을 여기에 저장합니다.
    attrs = {}

    # 각 입력 컬럼에 대해 StringField 추가
    for col in input_cols:
        attrs[col] = StringField(col, validators=[DataRequired()])

    # 제출 버튼 추가
    attrs['submit'] = SubmitField('예측')

    # type() 함수를 사용하여 동적으로 클래스를 생성합니다.
    # 첫 번째 인자: 클래스 이름 ('InferenceForm')
    # 두 번째 인자: 상속할 클래스의 튜플 (FlaskForm,)
    # 세 번째 인자: 클래스 속성을 담은 딕셔너리 (attrs)
    InferenceForm = type('InferenceForm', (FlaskForm,), attrs)

    return InferenceForm    

class RetrainForm(FlaskForm):
    """
    모델 재학습을 위한 폼 (주로 CSRF 보호용).
    사용자로부터 특정 입력이 필요하다면 여기에 필드를 추가하세요.
    예: epochs = IntegerField('에폭', default=10, validators=[DataRequired()])
    """
    submit = SubmitField('재학습 시작') # 재학습을 트리거하는 버튼