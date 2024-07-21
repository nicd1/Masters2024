from sklearn import svm
from sklearn import metrics

classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

Y.pred = classifier.predict(X_test)

print("Accuracy:" ,metrics.accuracy_score(Y_test, Y_pred))
print("Recall:" ,metrics.recall_score(Y_test, Y_pred))
print("Precision:" ,metrics.precision_score(Y_test, Y_pred))