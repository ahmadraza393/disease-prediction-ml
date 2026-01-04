from sklearn.metrics import accuracy_score

def evaluate_models(models, X, y):
    results = {}

    for name, model in models.items():
        preds = model.predict(X)
        results[name] = accuracy_score(y, preds)

    return results
