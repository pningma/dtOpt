import numpy as np
import pandas as pd
import pulp as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# raw = pd.read_sas('data/accepts.sas7bdat')
# raw.to_csv('data/accepts.csv', encoding='utf-8')
raw = pd.read_csv('data/accepts.csv')

raw.drop(['weight'], axis=1, inplace=True)
y = raw['bad']
X = raw.drop(['bad'], axis=1)
X.fillna(0, inplace=True)

le=LabelEncoder()
col_cat = X.columns[X.dtypes == 'object']
for c in col_cat:
    X[c] = le.fit_transform(X[c])

dtree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=123)
dtree.fit(X, y)
nodes = dtree.tree_.apply(X.to_numpy().astype(np.float32))
raw['_nodes_'] = nodes

nodes_agg = raw.groupby('_nodes_').agg(
    n_samp=('bad', np.size), 
    n_bad=('bad', np.sum),
    loan_amt=('loan_amt', np.sum)  
)

pl.listSolvers(onlyAvailable=True)
solver = pl.GLPK_CMD()

n = range(nodes_agg.shape[0])

prob = pl.LpProblem("Decison_Tree_Nodes_Selection", sense=pl.LpMaximize)
w = [pl.LpVariable("w"+str(i), 0, 1, pl.LpBinary) for i in n]
prob += pl.lpSum([list(nodes_agg.n_samp)[i] * w[i] for i in n])
prob += pl.lpSum([list(nodes_agg.n_bad)[i] * w[i] for i in n]) <= \
    pl.lpSum([list(nodes_agg.n_samp)[i] * w[i] for i in n]) * .1
prob += pl.lpSum([list(nodes_agg.loan_amt)[i] * w[i] for i in n]) <= 5e8

prob.writeLP("dtOpt.lp")
prob.solve(solver)
print("Status:", pl.LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)

