"""
Generates HANDWRITTEN.md: working, equations and a step-by-step hand calculation
for each of the six algorithms, on a 6-row sample of the dataset.
Every number in the guide is computed here, so the hand solution can be checked.
"""
import itertools
import numpy as np
from sklearn.datasets import load_diabetes

d = load_diabetes(as_frame=True, scaled=False)
S = d.frame.loc[[0, 1, 2, 3, 4, 5], ["bmi", "bp", "target"]].reset_index(drop=True)
x = S["bmi"].to_numpy(); x2 = S["bp"].to_numpy(); y = S["target"].to_numpy()
n = len(x)
q_bmi, q_bp = 27.0, 90.0          # query point used in every example
L = []
w = L.append
f = lambda v, k=2: f"{v:.{k}f}"

w("# Handwritten solutions: six regression algorithms\n")
w("Copy these by hand for the handwritten part. Every number below is produced by `handwritten_examples.py`, so you can check your working.\n")
w("## Sample used\n")
w("Six rows of the Diabetes dataset (raw values). `bmi` = body mass index, `bp` = average blood pressure, `y` = disease progression score.\n")
w("| Row | bmi (x1) | bp (x2) | y |\n|---|---|---|---|")
for i in range(n): w(f"| {i+1} | {x[i]} | {f(x2[i],0)} | {f(y[i],0)} |")
w(f"\nQuery point for every prediction: **bmi = {q_bmi}, bp = {f(q_bp,0)}**.\n")

# 1 ---------------------------------------------------------------- linear regression
xb, yb = x.mean(), y.mean()
sxy = ((x - xb) * (y - yb)).sum(); sxx = ((x - xb) ** 2).sum()
b1 = sxy / sxx; b0 = yb - b1 * xb
w("## 1. Linear Regression\n")
w("**Working.** Fit a straight line through the data that minimises the sum of squared errors.\n")
w("**Equations.**\n\n```\ny_hat = b0 + b1*x\nb1 = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)\nb0 = y_mean - b1*x_mean\n```\n")
w(f"**Steps (x = bmi).**\n\n1. x_mean = {f(xb,3)}, y_mean = {f(yb,3)}\n")
w("| x | y | x - x_mean | y - y_mean | product | (x - x_mean)^2 |\n|---|---|---|---|---|---|")
for i in range(n): w(f"| {x[i]} | {f(y[i],0)} | {f(x[i]-xb,3)} | {f(y[i]-yb,3)} | {f((x[i]-xb)*(y[i]-yb),3)} | {f((x[i]-xb)**2,3)} |")
w(f"\n2. Sum of products = {f(sxy,3)}, sum of squares = {f(sxx,3)}")
w(f"3. b1 = {f(sxy,3)} / {f(sxx,3)} = **{f(b1,4)}**")
w(f"4. b0 = {f(yb,3)} - {f(b1,4)} x {f(xb,3)} = **{f(b0,3)}**")
w(f"5. Prediction at bmi = {q_bmi}: y_hat = {f(b0,3)} + {f(b1,4)} x {q_bmi} = **{f(b0+b1*q_bmi)}**\n")

# 2 ---------------------------------------------------------------- decision tree
def sse(v): return ((v - v.mean()) ** 2).sum() if len(v) else 0.0
xs = np.sort(np.unique(x)); thr = (xs[:-1] + xs[1:]) / 2
w("## 2. Decision Tree Regression\n")
w("**Working.** Try every possible split point. Keep the one that leaves the smallest squared error inside the two groups. Each leaf predicts the mean of its rows.\n")
w("**Equations.**\n\n```\nSSE(group) = sum((y - mean(group))^2)\ncost(split) = SSE(left) + SSE(right)      choose the split with the lowest cost\nleaf prediction = mean of y in that leaf\n```\n")
w(f"**Steps (split on bmi).** SSE before any split = {f(sse(y))}\n")
w("| Split: bmi <= t | Left y values | Right y values | SSE left | SSE right | Total |\n|---|---|---|---|---|---|")
best = None
for t in thr:
    l, r = y[x <= t], y[x > t]; c = sse(l) + sse(r)
    if best is None or c < best[1]: best = (t, c, l.mean(), r.mean())
    w(f"| {f(t)} | {', '.join(f(v,0) for v in l)} | {', '.join(f(v,0) for v in r)} | {f(sse(l))} | {f(sse(r))} | {f(c)} |")
t, c, lm, rm = best
w(f"\nBest split: **bmi <= {f(t)}** (total SSE {f(c)}). Left leaf predicts {f(lm)}, right leaf predicts {f(rm)}.")
dt_pred = lm if q_bmi <= t else rm
w(f"Prediction at bmi = {q_bmi}: goes {'left' if q_bmi <= t else 'right'}, so y_hat = **{f(dt_pred)}**\n")

# 3 ---------------------------------------------------------------- random forest
def stump(xx, yy):
    u = np.sort(np.unique(xx)); b = None
    for tt in (u[:-1] + u[1:]) / 2:
        l, r = yy[xx <= tt], yy[xx > tt]; cc = sse(l) + sse(r)
        if b is None or cc < b[1]: b = (tt, cc, l.mean(), r.mean())
    return b
boots = [[0, 0, 2, 3, 4, 5], [1, 2, 2, 3, 5, 5], [0, 1, 1, 3, 4, 4]]
w("## 3. Random Forest Regression\n")
w("**Working.** Build many trees, each on a bootstrap sample (rows drawn with replacement), then average their predictions. Averaging reduces the variance of a single tree.\n")
w("**Equations.**\n\n```\ny_hat = (1/T) * sum of tree_t(x) for t = 1..T\n```\n")
w("**Steps (T = 3 one-split trees).**\n\n| Tree | Rows drawn | Best split | Left mean | Right mean | Prediction at query |\n|---|---|---|---|---|---|")
rf = []
for k, idx in enumerate(boots, 1):
    tt, cc, a, b = stump(x[idx], y[idx]); p = a if q_bmi <= tt else b; rf.append(p)
    w(f"| {k} | {', '.join(str(i+1) for i in idx)} | bmi <= {f(tt)} | {f(a)} | {f(b)} | {f(p)} |")
w(f"\nForest prediction = ({' + '.join(f(p) for p in rf)}) / 3 = **{f(np.mean(rf))}**\n")

# 4 ---------------------------------------------------------------- neural network
xm, xsd, ym, ysd = x.mean(), x.std(), y.mean(), y.std()
xi, yi = (x[0] - xm) / xsd, (y[0] - ym) / ysd
w1 = np.array([0.5, -0.4]); bb1 = np.array([0.1, 0.2]); w2 = np.array([0.6, -0.3]); bb2 = 0.05; lr = 0.1
z = w1 * xi + bb1; h = np.maximum(z, 0); out = (w2 * h).sum() + bb2; loss = 0.5 * (out - yi) ** 2
dout = out - yi; gw2 = dout * h; gb2 = dout; dh = dout * w2; dz = dh * (z > 0); gw1 = dz * xi; gb1 = dz
w("## 4. Neural Network Regression\n")
w("**Working.** Inputs pass through layers of weighted sums and activations (forward pass). The error is sent backwards to get the gradient of each weight (backpropagation), and every weight moves a small step against its gradient.\n")
w("**Equations.**\n\n```\nz = w1*x + b1          h = ReLU(z) = max(0, z)\ny_hat = w2 . h + b2    Loss = 0.5 * (y_hat - y)^2\nw_new = w - lr * dLoss/dw\n```\n")
w(f"**Steps (1 input, 2 hidden ReLU neurons, 1 output, one training step on row 1).**\n")
w(f"1. Scale the data (needed for gradient descent): x_s = (x - {f(xm,3)}) / {f(xsd,3)}, y_s = (y - {f(ym,3)}) / {f(ysd,3)}. Row 1 gives x_s = {f(xi,4)}, y_s = {f(yi,4)}.")
w(f"2. Starting weights: w1 = [{w1[0]}, {w1[1]}], b1 = [{bb1[0]}, {bb1[1]}], w2 = [{w2[0]}, {w2[1]}], b2 = {bb2}, lr = {lr}.")
w(f"3. Forward: z = [{f(z[0],4)}, {f(z[1],4)}], h = [{f(h[0],4)}, {f(h[1],4)}], y_hat = {f(out,4)}, Loss = {f(loss,4)}.")
w(f"4. Backward: dLoss/dy_hat = y_hat - y_s = {f(dout,4)}.")
w(f"   - dLoss/dw2 = [{f(gw2[0],4)}, {f(gw2[1],4)}], dLoss/db2 = {f(gb2,4)}")
w(f"   - dLoss/dz = [{f(dz[0],4)}, {f(dz[1],4)}] (zero where the ReLU was off)")
w(f"   - dLoss/dw1 = [{f(gw1[0],4)}, {f(gw1[1],4)}], dLoss/db1 = [{f(gb1[0],4)}, {f(gb1[1],4)}]")
nw1, nb1, nw2, nb2 = w1 - lr * gw1, bb1 - lr * gb1, w2 - lr * gw2, bb2 - lr * gb2
z2 = nw1 * xi + nb1; out2 = (nw2 * np.maximum(z2, 0)).sum() + nb2
w(f"5. Update: w1 = [{f(nw1[0],4)}, {f(nw1[1],4)}], b1 = [{f(nb1[0],4)}, {f(nb1[1],4)}], w2 = [{f(nw2[0],4)}, {f(nw2[1],4)}], b2 = {f(nb2,4)}.")
w(f"6. Check: new y_hat = {f(out2,4)}, new Loss = **{f(0.5*(out2-yi)**2,4)}** (was {f(loss,4)}, so the step reduced the error).\n")

# 5 ---------------------------------------------------------------- xgboost
lam, eta = 1.0, 0.3
base = y.mean(); g = base - y; hss = np.ones(n)
def score(G, H): return G * G / (H + lam)
w("## 5. XGBoost Regression\n")
w("**Working.** Start from one constant prediction. Each new tree is fitted to the errors of the model so far, and its output is added with a small learning rate. A regularisation term (lambda) shrinks the leaf values.\n")
w("**Equations (squared error loss).**\n\n```\ng_i = y_hat_i - y_i        h_i = 1\nleaf weight  w = -G / (H + lambda)             G, H = sums of g and h in the leaf\nGain = 0.5 * [ GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda) ] - gamma\ny_hat_new = y_hat_old + eta * w\n```\n")
w(f"**Steps (first tree, lambda = {lam}, gamma = 0, eta = {eta}).**\n")
w(f"1. Base prediction = mean of y = {f(base,3)}.")
w("2. Gradients g = base - y: " + ", ".join(f(v) for v in g) + f". Sum G = {f(g.sum(),3)}, H = {n}.")
w("3. Gain of each split on bmi:\n\n| Split: bmi <= t | GL | HL | GR | HR | Gain |\n|---|---|---|---|---|---|")
bx = None
for tt in thr:
    m = x <= tt; GL, HL, GR, HR = g[m].sum(), m.sum(), g[~m].sum(), (~m).sum()
    gain = 0.5 * (score(GL, HL) + score(GR, HR) - score(g.sum(), n))
    if bx is None or gain > bx[1]: bx = (tt, gain, GL, HL, GR, HR)
    w(f"| {f(tt)} | {f(GL)} | {HL} | {f(GR)} | {HR} | {f(gain)} |")
tt, gain, GL, HL, GR, HR = bx
wl, wr = -GL / (HL + lam), -GR / (HR + lam)
w(f"\n4. Best split: **bmi <= {f(tt)}** (gain {f(gain)}).")
w(f"5. Leaf weights: left = -({f(GL)}) / ({HL} + {lam}) = {f(wl,3)}, right = -({f(GR)}) / ({HR} + {lam}) = {f(wr,3)}.")
wq = wl if q_bmi <= tt else wr
w(f"6. Prediction at bmi = {q_bmi} after one tree: {f(base,3)} + {eta} x {f(wq,3)} = **{f(base+eta*wq)}**. More trees repeat steps 2 to 6 on the new errors.\n")

# 6 ---------------------------------------------------------------- knn
m1, s1, m2, s2 = x.mean(), x.std(), x2.mean(), x2.std()
dist = np.sqrt(((x - q_bmi) / s1) ** 2 + ((x2 - q_bp) / s2) ** 2); order = np.argsort(dist)
w("## 6. K-Nearest Neighbors Regression (KNN)\n")
w("**Working.** There is no training step. To predict, find the k rows closest to the query and average their y values. Features must be scaled first, otherwise the feature with the larger numbers dominates the distance.\n")
w("**Equations.**\n\n```\nscaled value = (x - mean) / std\nd = sqrt( (x1 - q1)^2 + (x2 - q2)^2 )     on scaled values\ny_hat = mean of y for the k nearest rows\n```\n")
w(f"**Steps (k = 3, features bmi and bp).** bmi: mean {f(m1,3)}, std {f(s1,3)}. bp: mean {f(m2,3)}, std {f(s2,3)}.\n")
w("| Row | bmi | bp | y | Scaled distance to query | Rank |\n|---|---|---|---|---|---|")
rank = {int(i): r + 1 for r, i in enumerate(order)}
for i in range(n): w(f"| {i+1} | {x[i]} | {f(x2[i],0)} | {f(y[i],0)} | {f(dist[i],3)} | {rank[i]} |")
nn = order[:3]
w(f"\nNearest 3 rows: {', '.join(str(int(i)+1) for i in nn)}. y_hat = ({' + '.join(f(y[i],0) for i in nn)}) / 3 = **{f(y[nn].mean())}**\n")

w("## Summary of the hand predictions\n")
w(f"| Algorithm | Prediction at bmi = {q_bmi}, bp = {f(q_bp,0)} |\n|---|---|")
for name, p in [("Linear Regression", b0 + b1 * q_bmi), ("Decision Tree (1 split)", dt_pred), ("Random Forest (3 trees)", np.mean(rf)),
                ("XGBoost (1 tree)", base + eta * wq), ("KNN (k = 3)", y[nn].mean())]:
    w(f"| {name} | {f(p)} |")
w("\nThe neural network example shows one training step on scaled data, so it is not in this table. These are tiny teaching examples on six rows. The real comparison on the full dataset is in `regression_comparison.py`.\n")
open("HANDWRITTEN.md", "w").write("\n".join(L))
print("\n".join(L)[:3800])
