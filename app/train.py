from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
