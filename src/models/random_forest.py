from sklearn.ensemble import RandomForestClassifier


def RandomForestModel(X_Train, Y_Train, X_Test):
    RandomForestclassifier = RandomForestClassifier(n_estimators=1000, min_samples_split=20, random_state=43)
    RandomForestclassifier.fit(X_Train, Y_Train)
    RFC_Prediction = RandomForestClassifier.predict(X_Test)


