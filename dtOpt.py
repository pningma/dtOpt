import numpy as np
import pandas as pd
import pulp as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

# raw = pd.read_sas('data/accepts.sas7bdat')
# raw.to_csv('data/accepts.csv', encoding='utf-8')
raw = pd.read_csv('data/accepts.csv')

raw.drop(['weight'], axis=1, inplace=True)
y = raw['bad']
X = raw.drop(['bad'], axis=1)
X.fillna(0, inplace=True)

le = LabelEncoder()
col_cat = X.columns[X.dtypes == 'object']
for c in col_cat:
    X[c] = le.fit_transform(X[c])

dtree = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=20, random_state=123)
dtree.fit(X, y)

dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=X.columns,
                                class_names="01",
                                filled=True, node_ids=True, proportion=True)

graph = graphviz.Source(dot_data, format="png")
graph

nodes = dtree.tree_.apply(X.to_numpy().astype(np.float32))
raw['_nodes_'] = nodes

nodes_agg = raw.groupby('_nodes_').agg(
    n_samp=('bad', np.size),
    n_bad=('bad', np.sum),
    loan_amt=('loan_amt', np.sum)
)

pl.listSolvers(onlyAvailable=True)
solver = pl.GLPK_CMD()

# n = range(nodes_agg.shape[0])

prob = pl.LpProblem("Decison_Tree_Nodes_Selection", sense=pl.LpMaximize)
w = [pl.LpVariable("w"+str(i), 0, 1, pl.LpBinary) \
    for i in nodes_agg.index]
prob += pl.lpSum([nodes_agg.n_samp[i] * w[j] \
    for j, i in enumerate(nodes_agg.index)
])
prob += pl.lpSum([nodes_agg.n_bad[i] * w[j] \
    for j, i in enumerate(nodes_agg.index)
]) <= pl.lpSum([nodes_agg.n_samp[i] * w[j] \
    for j, i in enumerate(nodes_agg.index)]) * .1
prob += pl.lpSum([nodes_agg.loan_amt[i] * w[j] \
    for j, i in enumerate(nodes_agg.index)]) <= 5e8

prob.writeLP("dtOpt.lp")
prob.solve(solver)
print("Status:", pl.LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)