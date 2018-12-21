import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel(r'analysis.xls',sheetname=0)

from sklearn import preprocessing
import numpy as np

le = preprocessing.LabelEncoder()
for i in df.columns.values[1:8]:
    le.fit(df[i])
    df[i] = le.transform(df[i])
    df[i] = df[i].astype('int')

data = df[df.columns[1:20]]
y = df['Price']
#y = df['DOM']
#y = y/(24*60*60*(10**9))

X_train, X_holdout, y_train, y_holdout = train_test_split(data.values, y, test_size=0.3,
random_state=17)

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

reg_tree = DecisionTreeRegressor(max_depth=3, random_state=17)
reg_tree.fit(X_train,y_train)
reg_tree_pred = reg_tree.predict(X_holdout)
reg_tree.score(X_holdout,y_holdout)

from sklearn.model_selection import GridSearchCV
tree_params = {'max_depth': range(2,100), 'max_features':range(2,18,2)}

locally_best_tree = GridSearchCV(reg_tree,tree_params,cv=5,n_jobs=-1,verbose=True)
locally_best_tree.fit(X_train,y_train) 
print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)
reg_tree = DecisionTreeRegressor(max_depth=7, max_features = 16, random_state=17)
reg_tree.fit(X_train,y_train)
reg_tree_pred = reg_tree.predict(X_holdout)
reg_tree.score(X_holdout,y_holdout)

i = reg_tree_pred - y_holdout
j = abs(i) <= y_holdout*0.3
#j = abs(i) <= 14
import numpy as np
np.shape(j)
sum(j)/np.shape(j)[0]

from sklearn.tree import export_graphviz
import graphviz 

#dot_data = StringIO()
dot_data = export_graphviz(reg_tree, feature_names = df.columns[1:20],out_file=None, filled=True)
graph = graphviz.Source(dot_data)
graph

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, max_depth = 18, max_features = 6,random_state=17)# you code here 
rf.fit(X_train,y_train)

tree_params = {'max_depth': range(2,20), 'max_features':range(2,18,2)}

locally_best_tree = GridSearchCV(rf,tree_params,cv=5,n_jobs=-1,verbose=True)
locally_best_tree.fit(X_train,y_train)

print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)

RF_tree_pred = rf.predict(X_holdout)
rf.score(X_holdout,y_holdout)

i = RF_tree_pred - y_holdout
j = abs(i) <= y_holdout*0.3
#j = abs(i) <= 21
import numpy as np
np.shape(j)
sum(j)/np.shape(j)[0]
list(RF_tree_pred)

from sklearn.tree import export_graphviz
import graphviz 

#dot_data = StringIO()
dot_data = export_graphviz(reg_tree, feature_names = df.columns[0:22],out_file=None, filled=True)
graph = graphviz.Source(dot_data)
graph