from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from constants.constants import RANDOM_STATE, N_NEIGHBORS



def get_model(name = 'knn'):
    """
    Get the model based on the name provided.
    Args:
        name (str): Name of the model to retrieve.
    Returns:
        model: The specified model.
    """
    if name == 'knn':
        return KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    elif name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(512, 128), random_state=RANDOM_STATE, activation='relu', solver='adam', max_iter=400)
    elif name == 'gbm':
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif name == 'svm':
        return SVC(kernel='rbf')
    elif name == 'xgb-binary':
        return XGBClassifier(eval_metric='logloss', objective='binary:logistic', random_state=RANDOM_STATE)
    elif name == 'lgbm-binary':
        return LGBMClassifier(objective='binary', verbose = -1 , random_state=RANDOM_STATE)
    elif name == 'xgb-multi':
        return XGBClassifier(eval_metric='mlogloss', objective='multi:softprob',num_class=4, random_state=RANDOM_STATE)
    elif name == 'lgbm-multi':
        return LGBMClassifier(objective='multiclass', num_class=4, verbose = -1 , random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Model {name} not recognized.")


def get_pipeline(name = 'knn'):
    """
    Get the pipeline for the specified model.
    Args:
        name (str): Name of the model to retrieve.
    Returns:
        pipeline: The specified model pipeline.
    """
    
    model = get_model(name)
    pipeline = make_pipeline(
        MinMaxScaler(),
        model
    )

    return pipeline


def train_model(pipeline, X_train, y_train):
    """
    Train the model using the provided training data.
    Args:
        model: The model to train.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
    Returns:
        model: The trained model.
    """
    pipeline.fit(X_train, y_train)
    return pipeline