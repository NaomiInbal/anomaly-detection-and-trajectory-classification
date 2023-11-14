from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import main

# In this file we create xgboost model, which is a basis for other models built on it.

def xgboost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=100, objective='binary:logistic', missing=1, seed=42, subsample=0.5,
                          learning_rate=0.1, max_depth=4, eval_metric='aucpr', early_stopping_rounds=10, )
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train, y_train, verbose=True,
              eval_set=[(X_test, y_test)])
    print(model)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    cf_matrix = confusion_matrix(y_test, y_pred)
    main.plot_confusion_matrix(cf_matrix)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # Data preparation
    data_frame = main.import_data()
    df_modified = main.data_normalization(data_frame)
    df_modified = main.group_tracks(df_modified)
    x, max_route_length = main.reshape_tracks(df_modified)
    y = main.read_y(df_modified)
    main.is_balanced_database(df_modified)
    X_train, X_test, y_train, y_test = main.encoder(x, y)
    # xgboost
    xgboost_model(X_train, X_test, y_train, y_test)