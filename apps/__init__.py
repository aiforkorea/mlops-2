# app/__init__.py
import os
from flask import Flask
from werkzeug.security import generate_password_hash
from .extensions import db, migrate, login_manager, csrf
from .config import Config

# 전역 변수/인스턴스 초기화 (extensions.py에서 정의)
def create_app():     #  factory 함수
    app = Flask(__name__)
    app.config.from_object(Config) # config.py에서 설정 로드
    # 확장 기능 초기화 연계
    db.init_app(app)                      # flask 앱에 db연결
    migrate.init_app(app,db)              # 없으면, flask db 명령어를 사용불가
    login_manager.init_app(app)  # flask 앱에 로그인 관리 연결
    csrf.init_app(app)                    # flask 앱에 CSRF 보호 연결 

    # Flask-Login: 사용자 로더 설정 (auth 블루프린트에서 import하여 사용)
    # create_app() 정의 또는 auth/__init__.py 정의하여 login_manager.user_loader 데코레이터와 함께 사용
    from apps.dbmodels import User  # User 모델 임포트
    @login_manager.user_loader
    def load_user(user_id):   # Flask-Login이 user_id를 기반으로 사용자 객체를 로드
        return User.query.get(int(user_id))
    # Flask-Login: Unauthorized Error 핸들링, login_view와 같은 기능이나, next 값 자동전달
    @login_manager.unauthorized_handler
    def unauthorized():
        """로그인되지 않은 사용자가 @login_required 페이지에 접근 시 redirect"""
        from flask import flash, redirect, url_for, request
        flash('로그인이 필요합니다.', 'warning')
        return redirect(url_for('auth.login', next=request.path))
    # 블루프린트 등록
    from .main import main
    from .auth import auth
    from .mlops import mlops

    app.register_blueprint(main)
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(mlops, url_prefix='/mlops')

    # db 테이블 생성 및 관리자 초기계정 생성
    with app.app_context():
        db.drop_all()         # 운영시에는 커멘트 처리 필요
        db.create_all()       # 테이블 생성

    """
        # 최초 관리자 계정 생성
        admin_username = app.config.get('ADMIN_USERNAME')
        admin_password = app.config.get('ADMIN_PASSWORD')

        if admin_username and admin_password:
            admin_user = User.query.filter_by(username=admin_username).first()
            if not admin_user:
                hashed_password = generate_password_hash(admin_password)
                new_admin = User(username=admin_username, password=hashed_password, is_admin=True)
                db.session.add(new_admin)
                db.session.commit()
                print(f"관리자 계정 '{admin_username}'이(가) 생성되었습니다.")
            else:
                print(f"관리자 계정 '{admin_username}'이(가) 이미 존재합니다.")
        else:
            print("ADMIN_USERNAME 또는 ADMIN_PASSWORD 환경 변수가 설정되지 않았습니다.")
    """
    return app
