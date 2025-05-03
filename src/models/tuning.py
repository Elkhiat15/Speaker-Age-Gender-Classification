from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from train import get_model

def get_param(name):
    if name == "xgb":
        # Define param grid for XGBoost
        param_grid = {
            "n_estimators": [200, 300 ,400],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [5, 7, 9, 11],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }

    elif name == "lgbm":
        # Define param grid for LightGBM
        param_grid = {
            "n_estimators": [200, 300, 350, 400],
            "learning_rate": [0.01, 0.1, 0.2, 0.4],
            "num_leaves": [50, 100, 120],
            "max_depth": [-1, 5, 10],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }

    return param_grid


def get_model_manual(name, params):
    if name == "xgb":
        # Create XGBoost model with the given parameters
        model = XGBClassifier(
                objective='multi:softmax', 
                num_class=4,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                **params
            )
    elif name == "lgbm":
        # Create LightGBM model with the given parameters
        model = LGBMClassifier(
            objective='multiclass',
            num_class=4, verbose=-1,
            random_state=42,
            **params)
        
    return model


def get_model_cv(name, params, n_iter):
    if name == "xgb":
        # Create XGBoost model with the given parameters
        xgb = get_model('xgb-multi')

        model = RandomizedSearchCV(
                    xgb,
                    param_distributions=params,
                    n_iter=n_iter,
                    cv=5,
                    scoring='accuracy',
                    verbose=1,
                    n_jobs=-1
                )
    elif name == "lgbm":
        # Create LightGBM model with the given parameters
        lgbm = get_model('lgbm-multi')
        model = RandomizedSearchCV(
            lgbm,
            param_distributions=params,
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )  

    return model



def manual_search(X_train, y_train, X_val, y_val, n_trials,  name):
        
    param_grid = get_param(name)
    model = get_model_manual(name, param_grid)

    param_sampler = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))

    best_model = None
    best_score = 0
    best_params = None

    for i, params in enumerate(param_sampler):
        print(f"Trial {i+1}/{n_trials} with params: {params}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        score = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {score:.4f}\n")

        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
            
    return best_model, best_params, best_score


def randomized_cv_search(X_train, y_train, name, n_iter=10):
    param_grid = get_param(name)
    model = get_model_cv(name, param_grid, n_iter)
    model.fit(X_train, y_train)
    print("Best LightGBM params:", model.best_params_)
    print("Best LightGBM score:", model.best_score_)
    best_model = model.best_estimator_
    return best_model

