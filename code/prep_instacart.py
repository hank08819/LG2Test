"""Supply chain at scale: Instacart Online Grocery Basket Analysis (3.4 million orders of 206,209 customers, Kaggle mirror of the
2017 Instacart release). Customer x product matrix of purchase counts over all orders (prior and train); the 2,000 products bought by
most customers are kept, and customers with at least 5 distinct of them; entries log(1 + count), rows scaled to unit length, so the
reference distance is the cosine distance between purchase profiles (the recipe of the Online Retail II store). Label: the department
of most of the customer's purchases among the kept products.
usage: python prep_instacart.py <csv dir> <out npz>"""
import sys, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
D, out = sys.argv[1:3]
orders = pd.read_csv(f"{D}/orders.csv", usecols=["order_id", "user_id"])
op = pd.concat([pd.read_csv(f"{D}/order_products__{s}.csv", usecols=["order_id", "product_id"]) for s in ("prior", "train")])
q = op.merge(orders, on="order_id").groupby(["user_id", "product_id"]).size().rename("n").reset_index(); print("lines", len(op), "pairs", len(q), flush=True)
top = q.groupby("product_id")["user_id"].nunique().sort_values(ascending=False).index[:2000]; q = q[q["product_id"].isin(top)]
keep = q.groupby("user_id")["product_id"].nunique(); keep = keep[keep >= 5].index; q = q[q["user_id"].isin(keep)]
users = np.sort(q["user_id"].unique()); prods = np.sort(top.values); ui = np.searchsorted(users, q["user_id"].values); pj = np.searchsorted(prods, q["product_id"].values)
X = coo_matrix((np.log1p(q["n"].values).astype(np.float32), (ui, pj)), shape=(len(users), len(prods))).toarray(); X /= np.linalg.norm(X, axis=1, keepdims=True)
pr = pd.read_csv(f"{D}/products.csv").set_index("product_id"); dep = pd.read_csv(f"{D}/departments.csv").set_index("department_id")["department"]
pdep = dep.loc[pr.loc[prods, "department_id"].values].values; names = pr.loc[prods, "product_name"].values.astype(str)
W = coo_matrix((q["n"].values.astype(np.float32), (ui, pj)), shape=X.shape).tocsr(); deps = np.unique(pdep)
M = np.stack([np.asarray(W[:, pdep == d].sum(1)).ravel() for d in deps], 1); lab = deps[M.argmax(1)]
np.savez(out, X=X.astype(np.float32), label=lab, names=names, user=users)
print("matrix", X.shape, "nonzero", float((X > 0).mean()), dict(zip(*np.unique(lab, return_counts=True))), flush=True)
