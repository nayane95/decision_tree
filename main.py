import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# import the dataset
dataset =  pd.read_csv("diabetes.csv")

# feature selection
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = dataset[feature_cols]
y = dataset.Outcome

# split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state=1)

# build model
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train, Y_train)

# predict
y_pred = classifier.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)
print(confusion_matrix(Y_test, y_pred))

# accuracy
accuracy=metrics.accuracy_score(Y_test,y_pred)
print("Accuracy:", accuracy)
# precision
precision = metrics.precision_score(Y_test, y_pred)
print("Precision score:",precision)
# recall
recall = metrics.recall_score(Y_test, y_pred)
print("Recall score:",recall)

# save classifier as an image
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,8), dpi=1000)
tree.plot_tree(classifier,
               feature_names = feature_cols,class_names=['0','1'],
               filled = True);
fig.savefig('dibetese_tree.png')

