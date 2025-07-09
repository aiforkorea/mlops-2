# apps/mlops/views.py
import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from apps import db
from apps.dbmodels import Dataset, InferenceRecord, MLResult, ModelInfo, PreprocessingInfo, User
from flask import current_app, json, render_template, flash, url_for, redirect, request
from flask_login import login_user, logout_user
from apps import mlops
from apps.ml_utils import create_model, dump_model, load_model  # views.py import해서 라우팅 등록
from . import mlops ## 추가
from .forms import AddModelForm, FeatureSelectionForm, FeatureTargetForm, TrainModelForm, UploadForm, ImputeForm
# 1. 데이터 업로드
@mlops.route('/', methods=['GET'])
def index():
    return render_template('mlops/base.html')
@mlops.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(current_app.config['UPLOAD_DIR'], file.filename)
            print(filepath)
            file.save(filepath)
            df = pd.read_csv(filepath)
            ds = Dataset(filename=file.filename, data=json.loads(df.to_json(orient='records')))
            db.session.add(ds)
            db.session.commit()
            flash("업로드 완료!")
            return redirect(url_for('mlops.list_ml'))
        else:
            flash("CSV 파일만 업로드 가능합니다.")
    return render_template('mlops/upload.html', form=form)
# 2. 데이터 리스트
@mlops.route('list', methods=['GET'])
def list_ml():
    datasets = Dataset.query.order_by(Dataset.id.desc()).all()
    return render_template('mlops/list_ml.html', datasets=datasets)

# 3. 데이터 전처리
@mlops.route('/preprocess/<int:dataset_id>', methods=['GET', 'POST'])
def preprocess(dataset_id):
    ds = Dataset.query.get(dataset_id)
    df = pd.DataFrame(ds.data)
    methods = ['mean','median']
    form = ImputeForm()
    form.impute_method.choices = [(m, m) for m in methods]  # (value, label) 튜플 리스트로 할당

    if form.validate_on_submit():
        method = form.impute_method.data
        if method == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif method == 'median':
            df = df.fillna(df.median(numeric_only=True))
        processed = PreprocessingInfo(
            dataset_id=ds.id,
            preprocessing_steps={'impute': method},
            processed_data=json.loads(df.to_json(orient='records'))
        )
        db.session.add(processed)
        db.session.commit()
        return redirect(url_for('mlops.select_features', preprocess_id=processed.id))
    return render_template('mlops/preprocess.html', df=df.head(10).to_html(), methods=methods, form=form)

# 4. 피처/타겟 선택
@mlops.route('/select_features/<int:preprocess_id>', methods=['GET', 'POST'])
def select_features(preprocess_id):
    preprocess = PreprocessingInfo.query.get_or_404(preprocess_id)
    df = pd.DataFrame(preprocess.processed_data)
    cols = df.columns.tolist()

    form = FeatureSelectionForm()
    form.features.choices = [(col, col) for col in cols]
    form.target.choices = [(col, col) for col in cols]

    if form.validate_on_submit():
        features = form.features.data
        target = form.target.data
        # Ensure the target is not included in the features
        if target in features:
            features.remove(target)
        if not features or not target:
            flash("피처와 타겟을 모두 선택해야 합니다.")
            return redirect(url_for('mlops.select_features', preprocess_id=preprocess_id))            
        print(features)
        print(target)
        return redirect(url_for(
            "mlops.train_model", 
            preprocess_id=preprocess_id, 
            features=",".join(features), 
            target=target
        ))
    return render_template('mlops/select_features.html', form=form)

"""
#@mlops.route('/select_features/<int:preprocess_id>', methods=['GET','POST'])
def select_features(preprocess_id):
    preprocess = PreprocessingInfo.query.get(preprocess_id)
    df = pd.DataFrame(preprocess.processed_data)
    cols = df.columns.tolist()
    if request.method == 'POST':
        features = request.form.getlist('features')
        target = request.form['target']
        return redirect(url_for("train_model", preprocess_id=preprocess_id, features=",".join(features), target=target))
    return render_template('select_features.html', columns=cols)
"""

# 5. 모델 추가(등록)
@mlops.route('/add_model', methods=['GET','POST'])
def add_model():
    model_types = ['RandomForest', 'LogisticRegression']
    form = AddModelForm()
    form.model_type.choices = [(mt, mt) for mt in model_types]

    if form.validate_on_submit():
        name = form.name.data
        mtype = form.model_type.data
        model = create_model(mtype)
        blob = dump_model(model)
        mi = ModelInfo(name=name, model_type=mtype, model_blob=blob)
        db.session.add(mi)
        db.session.commit()
        flash("모델이 등록되었습니다.")
        return redirect(url_for('mlops.add_model'))
    return render_template('mlops/add_model.html', form=form, models=ModelInfo.query.all())

# 6. 모델 훈련, 교차검증, 성능평가

@mlops.route('/train/<int:preprocess_id>', methods=['GET', 'POST'])
def train_model(preprocess_id):
    features = request.args.get('features', '').split(',')
    target = request.args.get('target')
    preprocess = PreprocessingInfo.query.get(preprocess_id)

    if not preprocess:
        flash("전처리 정보를 찾을 수 없습니다.")
        return redirect(url_for('some_error_route'))

    ds = Dataset.query.get(preprocess.dataset_id)
    if not ds:
        flash("데이터셋을 찾을 수 없습니다.")
        return redirect(url_for('some_error_route'))

    df = pd.DataFrame(preprocess.processed_data)
    X = df[features]
    y = df[target]

    models = ModelInfo.query.all()
    metrics = None
    selected_model_id = None

    # 폼 객체 및 모델 선택 리스트
    form = TrainModelForm()
    form.model_id.choices = [(m.id, f"{m.name} ({m.model_type})") for m in models]

    if form.validate_on_submit():
        selected_model_id = form.model_id.data
        model_info = ModelInfo.query.get(selected_model_id)

        if not model_info:
            flash("선택된 모델 정보를 찾을 수 없습니다.")
            return redirect(
                url_for('mlops.train_model', preprocess_id=preprocess_id, features=','.join(features), target=target)
            )

        model = load_model(model_info.model_blob)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 기본 평가지표
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
        }

        try:
            # cross_validate (roc_auc는 별도 아래에서 추가)
            results = cross_validate(model, X, y, cv=cv, scoring=scorers)
            metrics = {
                k.replace('test_', ''): float(np.mean(v))
                for k, v in results.items() if k.startswith('test_')
            }

            # === ROC AUC 별도 계산 ===
            # 1) y가 문자열이면 정수 인코딩
            y_np = np.array(y)
            if y_np.dtype.kind in 'OUS':  # 문자열, object, category
                encoder = LabelEncoder()
                y_enc = encoder.fit_transform(y_np)
            else:
                y_enc = y_np
            # 2) 확률 예측값 얻기
            y_proba = cross_val_predict(model, X, y_enc, cv=cv, method='predict_proba')
            n_classes = len(np.unique(y_enc))
            # 3) roc_auc_score 계산
            if n_classes > 2:
                roc_auc = roc_auc_score(y_enc, y_proba, average='weighted', multi_class='ovr')
            else:
                roc_auc = roc_auc_score(y_enc, y_proba[:, 1])
            metrics['roc_auc'] = float(roc_auc)
        except Exception as e:
            flash(f"모델 평가 중 오류: {e}")
            metrics = None

        if metrics:
            # 최종 전체 데이터로 fit 후 저장
            model.fit(X, y)
            model_info.model_blob = dump_model(model)
            db.session.commit()
            flash("모델 학습 및 저장이 완료되었습니다.")

            # 결과 저장
            ml_result = MLResult(
                model_id=selected_model_id,
                dataset_id=ds.id,
                features=features,
                target=target,
                metrics=metrics
            )
            db.session.add(ml_result)
            db.session.commit()
            flash("모델 학습 결과가 저장되었습니다.")

    return render_template(
        'mlops/train_model.html',
        form=form,
        models=models,
        features=features,
        target=target,
        preprocess_id=preprocess_id,
        metrics=metrics
    )

# 7. 추론 API (입력폼)
@mlops.route('infer/<int:model_id>', methods=['GET','POST'])
def model_infer(model_id):
    model_info = ModelInfo.query.get(model_id)
    # 마지막으로 평가한 MLResult에서 feature 정보 가져옴
    last_result = MLResult.query.filter_by(model_id=model_id).order_by(MLResult.id.desc()).first()
    if not last_result:
        flash("모델에 사용된 피처 정보가 없습니다.")
        return redirect(url_for('mlops.list_ml'))
    input_cols = last_result.features
    pred_result = None
    inference_rec = None
    if request.method == 'POST':
        values = {col: request.form[col] for col in input_cols}
        model = load_model(model_info.model_blob)
        X = pd.DataFrame([values])
        for col in input_cols:  # float 변환 시도
            try: X[col] = X[col].astype(float)
            except: pass
        prediction = model.predict(X)[0]
        pred_result = prediction
        # 기록으로 저장
        inference_rec = InferenceRecord(
            model_id=model_info.id,
            input_data=values,
            output_data={'prediction':prediction}
        )
        db.session.add(inference_rec)
        db.session.commit()
    return render_template('mlops/inference.html',
        input_cols=input_cols, pred_result=pred_result, inference_rec=inference_rec, model_id=model_id)

# 8. 추론 피드백(정답 DB 저장)
@mlops.route('confirm_inference/<int:inference_id>', methods=['POST'])
def confirm_inference(inference_id):
    rec = InferenceRecord.query.get(inference_id)
    actual = request.form['actual']
    rec.is_confirmed = True
    rec.confirmed_data = {'actual': actual}
    db.session.commit()
    flash("정답이 저장되었습니다. 재학습 시 활용됩니다.")
    return redirect(url_for('mlops.model_infer', model_id=rec.model_id))

# 9. 추가된 데이터로 재학습 (활용)
@mlops.route('/retrain/<int:model_id>', methods=['GET','POST'])
def retrain(model_id):
    model_info = ModelInfo.query.get(model_id)
    # 최근 MLResult에서 features, target 추출
    last_result = MLResult.query.filter_by(model_id=model_id).order_by(MLResult.id.desc()).first()
    if not last_result:
        flash("이 모델에 학습된 결과정보가 없습니다! 학습 먼저 하세요.")
        return redirect(url_for('mlops.add_model'))
    features = last_result.features
    target = last_result.target
    # 추론 후 정답까지 입력된 레코드
    recs = InferenceRecord.query.filter_by(model_id=model_id, is_confirmed=True).all()
    retrain_report = None
    if request.method == 'POST' and len(recs) > 0:
        X_new = pd.DataFrame([r.input_data for r in recs])
        y_new = [r.confirmed_data['actual'] for r in recs]
        # 기존 학습 데이터도 추가(오리지널 데이터)
        preprocess_id = MLResult.query.filter_by(model_id=model_id).order_by(MLResult.id.desc()).first()
        mlr = MLResult.query.get(preprocess_id.id)
        ds = Dataset.query.get(mlr.dataset_id)
        orig_df = pd.DataFrame(ds.data)
        orig_X = orig_df[features]
        orig_y = orig_df[target]
        X_total = pd.concat([orig_X,X_new],axis=0)
        y_total = pd.concat([orig_y,pd.Series(y_new)],axis=0)
        model = create_model(model_info.model_type)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
        }
        try:
            results = cross_validate(model, X_total, y_total, cv=cv, scoring=scorers)
            metrics = {k.replace('test_',''):float(np.mean(v)) for k,v in results.items() if k.startswith('test_')}
            model.fit(X_total, y_total)
            model_info.model_blob = dump_model(model)
            db.session.commit()
            retrain_report = metrics
        except Exception as e:
            flash(f"재학습 중 오류: {e}")
            retrain_report = None
    else:
        retrain_report = None
    return render_template('mlops/train_model.html',
        retrain_report=retrain_report, retrain_mode=True, model_id=model_id)
