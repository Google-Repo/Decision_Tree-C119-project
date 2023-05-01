import pandas as pd

#Column Name
col_names = ['passengerId','pclass','sex','age','sibsp','parch','survived']

df = pd.read_csv("titanic.csv", names=col_names).iloc[1:]

print(df.head())

Datas = ['passengerId','pclass','sex','age','sibsp','parch']
x=df[Datas]
y = df.survived

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#splitting data in training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3,random_state=1)

#intialising the Decision Tree Model
clf=DecisionTreeClassifier()

#Fitting the data into the model
clf = clf.fit(x_train, y_train)

#Calculating the accuracy of the model
y_pred = clf.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO() #Where we will store the data from our decision tree classifier as text.

#using export_grapviz function to create a graph representation of the decision tree which can be written in out file.
export_graphviz(clf,out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=Datas, class_names=['0','1'])

print(dot_data.getvalue())

clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Where we will store the data from our decision tree classifier as text.
dot_data = StringIO()


# using export_graphviz function to create a graphviz representation of the decision tree
dot_data = StringIO() #Where we will store the data from our decision tree classifier as text.


# using export_graphviz function to create a graphviz representation of the decision tree
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=Datas, class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('titanic.png')
Image(graph.create_png())
