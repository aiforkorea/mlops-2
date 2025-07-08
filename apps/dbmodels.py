#apps/dbmodels.py
import enum
import uuid
from flask import current_app
from sqlalchemy import JSON
from apps import db  # apps에서 db를 import
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash 
#hash import
from datetime import datetime
from apps.config import Config

class User(db.Model, UserMixin):  # db.Model, UserMixin 상속하는 User 클래스 생성
    __tablename__= "users"  # 삭제시, 테이블 이름은 Model 이름 소문자 
    id=db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String, index=True)
    email=db.Column(db.String, unique=True, index= True)
    password_hash=db.Column(db.String)
    is_admin=db.Column(db.Boolean,default=False)
    is_active=db.Column(db.Boolean, default=True)  # 활성화 여부
    usage_count = db.Column(db.Integer, default=0)
    # 특정 User에 대한 일일/월간 사용량 제한을 위한 필드 추가 가능 (예: daily_limit, monthly_limit)
    daily_limit = db.Column(db.Integer, default=1000)
    monthly_limit = db.Column(db.Integer, default=5000)
    created_at=db.Column(db.DateTime, default= datetime.now)
    updated_at=db.Column(db.DateTime, default= datetime.now, onupdate=datetime.now)
    #api_keys = db.relationship('APIKey', backref='user', lazy=True, cascade='all, delete-orphan')
    #usage_logs = db.relationship('UsageLog', backref='user', lazy=True, cascade='all, delete-orphan')
    #iris_results = db.relationship('IRIS', backref='user', lazy=True, cascade='all, delete-orphan')

    # The 'password' property and its setter
    @property
    def password(self):
        """
        Prevent direct reading of the password.
        Attempting to read `user.password` will raise an AttributeError.
        """
        raise AttributeError('password is not a readable attribute')
    @password.setter
    def password(self, password):
        """
        Hashes the plain-text password and stores it in password_hash.
        This allows you to do `user.password = "mysecretpassword"`.
        """
        self.password_hash = generate_password_hash(password)
    # Method to check password (you likely already have this or check_password)
    def check_password(self, password):
        """Checks if the provided password matches the stored hash."""
        if self.password_hash is None:
            return False
        return check_password_hash(self.password_hash, password)
    # 비밀번호 체크
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    # Flask-Login required methods
    @property
    def is_authenticated(self):
        return True
    @property
    def is_anonymous(self):
        return False
    def __repr__(self):
        return f'<User {self.username}>'
    # 이메일 중복 체크
    def is_duplicate_email(self):
        return User.query.filter_by(email=self.email).first() is not None
    # flask-login에서 사용자 ID를 문자열로 반환하기 위한 함수
    # 이 함수는 Flask-Login이 사용자 객체를 로드할 때 사용됨
    def get_id(self):
        return str(self.id)
    # 현재 로그인하고 있는 사용자 정보 취득 함수 설정은 apps/__init__.py에서 정의함

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128), nullable=False)
    data = db.Column(JSON)  # list of dicts
    created_at = db.Column(db.DateTime, default=datetime.now)

class PreprocessingInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    preprocessing_steps = db.Column(JSON)
    processed_data = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.now)

class ModelInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    model_type = db.Column(db.String(64), nullable=False)  # e.g. 'RandomForest'
    model_blob = db.Column(db.LargeBinary, nullable=True)  # pickle
    created_at = db.Column(db.DateTime, default=datetime.now)

class MLResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model_info.id'))
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    features = db.Column(JSON)
    target = db.Column(db.String(64))
    metrics = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.now)

class InferenceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('model_info.id'))
    input_data = db.Column(JSON)
    output_data = db.Column(JSON)
    is_confirmed = db.Column(db.Boolean, default=False)
    confirmed_data = db.Column(JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
