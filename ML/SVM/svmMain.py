from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Assuming 'X' is your input features and 'y' is your target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

Y.pred = classifier.predict(X_test)

print("Accuracy:" ,metrics.accuracy_score(Y_test, Y_pred))
print("Recall:" ,metrics.recall_score(Y_test, Y_pred))
print("Precision:" ,metrics.precision_score(Y_test, Y_pred))