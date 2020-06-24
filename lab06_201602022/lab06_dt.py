import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydot


def data_processing(filename):
	data = pd.read_csv(filename)
	#test = pd.get_dummies(data['Embarked'])
	data = data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
	#data = pd.concat([data, test], axis=1)

	le = LabelEncoder()
	le.fit(['male', 'female'])
	data['Sex'] = le.transform(data['Sex'])
	
	data['Age'].fillna(data['Age'].mean(), inplace=True)
	data['Fare'].fillna(data['Fare'].mean(), inplace=True)
	
	return data


def fit_and_predict(X_train, Y_train, X_test):
	model = DecisionTreeClassifier(max_depth=10, random_state=42)
	model.fit(X_train, Y_train)
	score = model.score(X_train, Y_train)
	save_graph(model)
	print('Accuracy :', score)
	prediction = model.predict(X_test)
	return prediction


def save_graph(dt):
	export_graphviz(
		dt, 
		out_file='dt.dot', 
		class_names=['No', 'Yes'], 
		feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', ''' 'C', 'Q', 'S' '''],
		impurity=False, 
		filled=True)
	(graph, ) = pydot.graph_from_dot_file('dt.dot', encoding='utf8')
	graph.write_png('dt.png')
	

def main():
	train_data = data_processing('train.csv')
	Y_train = train_data['Survived']
	X_train = train_data.drop(['Survived'], axis=1)
	X_test = data_processing('test.csv')

	prediction = fit_and_predict(X_train, Y_train, X_test)

	result = pd.DataFrame({
		"PassengerId": X_test["PassengerId"],
		"Survived": prediction
	})
	result.to_csv('result.csv', index=False)
	

if __name__=='__main__':
	main()

