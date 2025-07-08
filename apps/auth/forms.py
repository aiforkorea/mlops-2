# apps/auth/forms.py 
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField  #구성요소
from wtforms.validators import DataRequired, Email, length  #유효성

# form class
class SignUpForm(FlaskForm):      # FlaskForm 상속
    username=StringField("사용자명", 
        validators=[ 
            DataRequired(message="사용자이름 필수"),
            length(max=30,message="30문자이내 입력"),
        ], 
    )
    email=StringField("이메일주소",     
        validators=[ 
            DataRequired(message="메일 주소 필수"),
            #Email(message="메일주소 형식 준수"),
        ], 
    )
    password=PasswordField("비밀번호", 
        validators=[ DataRequired(message="비밀번호 필수")]  
    )
    submit=SubmitField("신규등록")
# form class 추가 - SignUpForm 클래스와 유사
class LoginForm(FlaskForm):      # FlaskForm 상속
    email=StringField("이메일주소", 
        validators=[ DataRequired(message="메일 주소 필수"),
        #Email(message="메일주소 형식 준수"), #email_validator 설치필수
    ], )
    password=PasswordField("비밀번호", 
        validators=[ DataRequired(message="비밀번호 필수")]  )
    submit=SubmitField("로그인")
