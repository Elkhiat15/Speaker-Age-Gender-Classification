from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score, 
    precision_score , 
    recall_score 
    )


def evaluate_binary(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    confusion = confusion_matrix(y, y_pred)

    if type == 0:
        print("\n\nnTRAINING TESTING RESULTS: \n===============================")
    elif type == 1:
        print("\n\VALIDATION RESULTS: \n===============================")
    elif type == 2:
        print("\n\nTESTING RESULTS: \n===============================")
    else:
        print("\n\nPlease choose a value from 0:2 : \n===============================")
        pass
    
    print(f"CONFUSION MATRIX:\n{confusion}")
    print(f"ACCURACY SCORE:\n{acc:.4f}")
    print("precision score:", round(precision_score(y,y_pred),2))
    print("Recall Accuracy:", round(recall_score(y,y_pred),2))
    print("Area Under Curve AUC:", round(roc_auc_score(y,y_pred),2))




def evaluate_multi(model, X, y, type = 0):
    y_pred = model.predict(X)
    acc = accuracy_score(y,y_pred)
    confusion = confusion_matrix(y, y_pred)

    if type == 0:
        print("\n\nnTRAINING TESTING RESULTS: \n===============================")
    elif type == 1:
        print("\n\VALIDATION RESULTS: \n===============================")
    elif type == 2:
        print("\n\nTESTING RESULTS: \n===============================")
    else:
        print("\n\nPlease choose a value from 0:2 : \n===============================")
        pass

    print(f"CONFUSION MATRIX:\n{confusion}")
    print(f"ACCURACY SCORE:\n{acc:.4f}")