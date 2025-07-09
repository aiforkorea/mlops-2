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
from .forms import AddModelForm, FeatureSelectionForm, FeatureTargetForm, RetrainForm, TrainModelForm, UploadForm, ImputeForm, create_inference_form
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
    last_result = MLResult.query.filter_by(model_id=model_id).order_by(MLResult.id.desc()).first()

    if not last_result:
        flash("모델에 사용된 피처 정보가 없습니다.")
        return redirect(url_for('mlops.list_ml'))

    input_cols = last_result.features

    # Create the form instance
    InferenceForm = create_inference_form(input_cols)
    form = InferenceForm()

    pred_result = None
    inference_rec = None

    if form.validate_on_submit(): # WTForms handles POST and validation
        values = {col: request.form[col] for col in input_cols} # Or use form.data for cleaned data if fields are typed
        model = load_model(model_info.model_blob)
        X = pd.DataFrame([values])
        for col in input_cols:
            try:
                X[col] = X[col].astype(float)
            except ValueError:
                # Handle cases where conversion to float fails (e.g., non-numeric input)
                flash(f"'{col}' 값은 유효한 숫자가 아닙니다. 다시 확인해주세요.", "error")
                return render_template('mlops/inference.html',
                                       form=form, # Pass the form back to re-render
                                       input_cols=input_cols,
                                       pred_result=pred_result,
                                       inference_rec=inference_rec,
                                       model_id=model_id)

        prediction = model.predict(X)[0]

        # --- 이 부분에 변경 사항 추가 ---
        # prediction 값을 표준 파이썬 int 또는 float로 변환합니다.
        if isinstance(prediction, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64)):
            prediction = int(prediction)
        elif isinstance(prediction, (np.float_, np.float16, np.float32, np.float64)):
            prediction = float(prediction)
        # 만약 예측 결과가 리스트/배열 형태이고 그 안에 NumPy 타입이 있다면, 리스트 내의 모든 요소를 변환해야 합니다.
        elif isinstance(prediction, np.ndarray):
            prediction = prediction.tolist() # NumPy 배열을 파이썬 리스트로 변환

            # 만약 리스트 안에 NumPy 타입이 남아있다면, 리스트 컴프리헨션으로 처리
            if prediction and any(isinstance(x, (np.generic)) for x in prediction):
                prediction = [
                    int(x) if isinstance(x, (np.int_, np.integer)) else
                    float(x) if isinstance(x, (np.float_, np.floating)) else x
                    for x in prediction
                ]

        # --- 변경 사항 끝 ---
        
        pred_result = prediction

        inference_rec = InferenceRecord(
            model_id=model_info.id,
            input_data=values,
            output_data={'prediction':prediction}
        )
        db.session.add(inference_rec)
        db.session.commit()

    return render_template('mlops/inference.html',
                           form=form, # Pass the form to the template
                           input_cols=input_cols,
                           pred_result=pred_result,
                           inference_rec=inference_rec,
                           model_id=model_id)

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

    # 폼 인스턴스 생성
    form = RetrainForm() # RetrainForm 인스턴스 생성

    if not last_result:
        flash("이 모델에 학습된 결과정보가 없습니다! 학습 먼저 하세요.", "warning") # 메시지 중요도
        # 폼이 없으므로, flash만 하고 바로 리다이렉트
        return redirect(url_for('mlops.add_model'))

    features = last_result.features
    target = last_result.target

    # 추론 후 정답까지 입력된 레코드
    recs = InferenceRecord.query.filter_by(model_id=model_id, is_confirmed=True).all()
    retrain_report = None

    # POST 요청이고, 폼이 유효하며, 추가된 데이터가 있을 경우에만 재학습 진행
    if form.validate_on_submit() and len(recs) > 0:
        # X_new의 데이터 타입을 확인하고, 필요한 경우 숫자로 변환
        # input_data는 딕셔너리 형태이므로 DataFrame 생성 시 주의
        X_new_raw = [r.input_data for r in recs]
        X_new = pd.DataFrame(X_new_raw)

        # 모든 피처가 숫자로 변환 가능한지 확인하고 변환 (inference.py와 동일하게)
        for col in features: # features를 기준으로 변환
            if col in X_new.columns:
                try:
                    X_new[col] = X_new[col].astype(float)
                except ValueError:
                    flash(f"추가된 데이터의 '{col}' 컬럼에 유효하지 않은 값이 있어 재학습을 중단합니다.", "error")
                    return render_template('mlops/train_model.html',
                                           form=form, # 폼을 다시 전달
                                           retrain_report=None,
                                           retrain_mode=True,
                                           model_id=model_id)
            else:
                # 만약 새로운 데이터에 원래 피처가 없다면, 0 또는 NaN으로 채울 수 있습니다.
                # 이는 데이터 전처리 전략에 따라 달라질 수 있습니다.
                X_new[col] = 0 # 예시: 없는 피처는 0으로 채움
                # flash(f"경고: 추가된 데이터에 '{col}' 피처가 없습니다. 0으로 처리됩니다.", "warning")

        # y_new도 적절한 타입으로 변환 (숫자형으로 가정)
        y_new_raw = [r.confirmed_data['actual'] for r in recs]
        try:
            y_new = pd.Series([float(y) for y in y_new_raw]) # 실제값도 숫자로 변환
        except ValueError:
            flash("추가된 정답 데이터에 유효하지 않은 값이 있어 재학습을 중단합니다.", "error")
            return render_template('mlops/train_model.html',
                                   form=form,
                                   retrain_report=None,
                                   retrain_mode=True,
                                   model_id=model_id)

        # 기존 학습 데이터도 추가(오리지널 데이터)
        # MLResult에서 dataset_id를 가져오는 부분이 잘못되었습니다.
        # last_result.dataset_id를 직접 사용해야 합니다.
        ds = Dataset.query.get(last_result.dataset_id)
        if not ds:
            flash("오리지널 데이터셋을 찾을 수 없어 재학습을 진행할 수 없습니다.", "error")
            return render_template('mlops/train_model.html',
                                   form=form,
                                   retrain_report=None,
                                   retrain_mode=True,
                                   model_id=model_id)

        orig_df = pd.DataFrame(ds.data)

        # 오리지널 데이터의 피처와 타겟도 적절한 타입으로 변환
        orig_X = orig_df[features].astype(float) # 원본 X도 float으로 통일
        orig_y = orig_df[target].astype(float) # 원본 y도 float으로 통일 (분류 타겟이라면 int)

        # 컬럼 순서를 맞추기 위해 X_new를 orig_X와 동일한 컬럼 순서로 재정렬
        X_new = X_new[features]

        X_total = pd.concat([orig_X, X_new], axis=0).reset_index(drop=True)
        y_total = pd.concat([orig_y, y_new], axis=0).reset_index(drop=True)

        # 모델 생성 및 재학습
        model = create_model(model_info.model_type)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            # 이진 분류가 아닌 다중 클래스 분류라면 roc_auc_score의 multi_class와 average 설정 확인 필요
            # needs_proba=True 또는 needs_threshold=True 설정에 따라 모델의 predict_proba 또는 decision_function 메소드가 필요함
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr') # 'ovr' (One-vs-Rest) 또는 'ovo' (One-vs-One)
        }
        try:
            results = cross_validate(model, X_total, y_total, cv=cv, scoring=scorers)
            metrics = {k.replace('test_',''):float(np.mean(v)) for k,v in results.items() if k.startswith('test_')}
            model.fit(X_total, y_total) # 최종 모델 학습
            model_info.model_blob = dump_model(model) # 새로 학습된 모델 저장
            db.session.commit()
            retrain_report = metrics
            flash("모델이 성공적으로 재학습되었습니다!", "success")
        except Exception as e:
            flash(f"재학습 중 오류가 발생했습니다: {e}", "error")
            retrain_report = None
    elif request.method == 'POST' and len(recs) == 0:
        flash("재학습을 위한 피드백 데이터가 없습니다. 추론 후 정답을 저장해주세요.", "info")
        retrain_report = None
    else: # GET 요청 시
        retrain_report = None

    return render_template('mlops/train_model.html',
        form=form, # <-- form 객체를 템플릿으로 전달합니다.
        retrain_report=retrain_report,
        retrain_mode=True, # 이 플래그는 템플릿에서 재학습 모드임을 나타낼 때 유용합니다.
        model_id=model_id)