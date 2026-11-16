# Handwritten solutions: six regression algorithms

Copy these by hand for the handwritten part. Every number below is produced by `handwritten_examples.py`, so you can check your working.

## Sample used

Six rows of the Diabetes dataset (raw values). `bmi` = body mass index, `bp` = average blood pressure, `y` = disease progression score.

| Row | bmi (x1) | bp (x2) | y |
|---|---|---|---|
| 1 | 32.1 | 101 | 151 |
| 2 | 21.6 | 87 | 75 |
| 3 | 30.5 | 93 | 141 |
| 4 | 25.3 | 84 | 206 |
| 5 | 23.0 | 101 | 135 |
| 6 | 22.6 | 89 | 97 |

Query point for every prediction: **bmi = 27.0, bp = 90**.

## 1. Linear Regression

**Working.** Fit a straight line through the data that minimises the sum of squared errors.

**Equations.**

```
y_hat = b0 + b1*x
b1 = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
b0 = y_mean - b1*x_mean
```

**Steps (x = bmi).**

1. x_mean = 25.850, y_mean = 134.167

| x | y | x - x_mean | y - y_mean | product | (x - x_mean)^2 |
|---|---|---|---|---|---|
| 32.1 | 151 | 6.250 | 16.833 | 105.208 | 39.063 |
| 21.6 | 75 | -4.250 | -59.167 | 251.458 | 18.062 |
| 30.5 | 141 | 4.650 | 6.833 | 31.775 | 21.623 |
| 25.3 | 206 | -0.550 | 71.833 | -39.508 | 0.302 |
| 23.0 | 135 | -2.850 | 0.833 | -2.375 | 8.122 |
| 22.6 | 97 | -3.250 | -37.167 | 120.792 | 10.562 |

2. Sum of products = 467.350, sum of squares = 97.735
3. b1 = 467.350 / 97.735 = **4.7818**
4. b0 = 134.167 - 4.7818 x 25.850 = **10.557**
5. Prediction at bmi = 27.0: y_hat = 10.557 + 4.7818 x 27.0 = **139.67**

## 2. Decision Tree Regression

**Working.** Try every possible split point. Keep the one that leaves the smallest squared error inside the two groups. Each leaf predicts the mean of its rows.

**Equations.**

```
SSE(group) = sum((y - mean(group))^2)
cost(split) = SSE(left) + SSE(right)      choose the split with the lowest cost
leaf prediction = mean of y in that leaf
```

**Steps (split on bmi).** SSE before any split = 10372.83

| Split: bmi <= t | Left y values | Right y values | SSE left | SSE right | Total |
|---|---|---|---|---|---|
| 22.10 | 75 | 151, 141, 206, 135, 97 | 0.00 | 6172.00 | 6172.00 |
| 22.80 | 75, 97 | 151, 141, 206, 135 | 242.00 | 3170.75 | 3412.75 |
| 24.15 | 75, 135, 97 | 151, 141, 206 | 1842.67 | 2450.00 | 4292.67 |
| 27.90 | 75, 206, 135, 97 | 151, 141 | 9902.75 | 50.00 | 9952.75 |
| 31.30 | 75, 141, 206, 135, 97 | 151 | 10032.80 | 0.00 | 10032.80 |

Best split: **bmi <= 22.80** (total SSE 3412.75). Left leaf predicts 86.00, right leaf predicts 158.25.
Prediction at bmi = 27.0: goes right, so y_hat = **158.25**

## 3. Random Forest Regression

**Working.** Build many trees, each on a bootstrap sample (rows drawn with replacement), then average their predictions. Averaging reduces the variance of a single tree.

**Equations.**

```
y_hat = (1/T) * sum of tree_t(x) for t = 1..T
```

**Steps (T = 3 one-split trees).**

| Tree | Rows drawn | Best split | Left mean | Right mean | Prediction at query |
|---|---|---|---|---|---|
| 1 | 1, 1, 3, 4, 5, 6 | bmi <= 22.80 | 97.00 | 156.80 | 156.80 |
| 2 | 2, 3, 3, 4, 6, 6 | bmi <= 23.95 | 89.67 | 162.67 | 162.67 |
| 3 | 1, 2, 2, 4, 5, 5 | bmi <= 22.30 | 75.00 | 156.75 | 156.75 |

Forest prediction = (156.80 + 162.67 + 156.75) / 3 = **158.74**

## 4. Neural Network Regression

**Working.** Inputs pass through layers of weighted sums and activations (forward pass). The error is sent backwards to get the gradient of each weight (backpropagation), and every weight moves a small step against its gradient.

**Equations.**

```
z = w1*x + b1          h = ReLU(z) = max(0, z)
y_hat = w2 . h + b2    Loss = 0.5 * (y_hat - y)^2
w_new = w - lr * dLoss/dw
```

**Steps (1 input, 2 hidden ReLU neurons, 1 output, one training step on row 1).**

1. Scale the data (needed for gradient descent): x_s = (x - 25.850) / 4.036, y_s = (y - 134.167) / 41.579. Row 1 gives x_s = 1.5486, y_s = 0.4049.
2. Starting weights: w1 = [0.5, -0.4], b1 = [0.1, 0.2], w2 = [0.6, -0.3], b2 = 0.05, lr = 0.1.
3. Forward: z = [0.8743, -0.4194], h = [0.8743, 0.0000], y_hat = 0.5746, Loss = 0.0144.
4. Backward: dLoss/dy_hat = y_hat - y_s = 0.1697.
   - dLoss/dw2 = [0.1484, 0.0000], dLoss/db2 = 0.1697
   - dLoss/dz = [0.1018, -0.0000] (zero where the ReLU was off)
   - dLoss/dw1 = [0.1577, -0.0000], dLoss/db1 = [0.1018, -0.0000]
5. Update: w1 = [0.4842, -0.4000], b1 = [0.0898, 0.2000], w2 = [0.5852, -0.3000], b2 = 0.0330.
6. Check: new y_hat = 0.5244, new Loss = **0.0071** (was 0.0144, so the step reduced the error).

## 5. XGBoost Regression

**Working.** Start from one constant prediction. Each new tree is fitted to the errors of the model so far, and its output is added with a small learning rate. A regularisation term (lambda) shrinks the leaf values.

**Equations (squared error loss).**

```
g_i = y_hat_i - y_i        h_i = 1
leaf weight  w = -G / (H + lambda)             G, H = sums of g and h in the leaf
Gain = 0.5 * [ GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda) ] - gamma
y_hat_new = y_hat_old + eta * w
```

**Steps (first tree, lambda = 1.0, gamma = 0, eta = 0.3).**

1. Base prediction = mean of y = 134.167.
2. Gradients g = base - y: -16.83, 59.17, -6.83, -71.83, -0.83, 37.17. Sum G = -0.000, H = 6.
3. Gain of each split on bmi:

| Split: bmi <= t | GL | HL | GR | HR | Gain |
|---|---|---|---|---|---|
| 22.10 | 59.17 | 1 | -59.17 | 5 | 1166.90 |
| 22.80 | 96.33 | 2 | -96.33 | 4 | 2474.70 |
| 24.15 | 95.50 | 3 | -95.50 | 3 | 2280.06 |
| 27.90 | 23.67 | 4 | -23.67 | 2 | 149.36 |
| 31.30 | 16.83 | 5 | -16.83 | 1 | 94.45 |

4. Best split: **bmi <= 22.80** (gain 2474.70).
5. Leaf weights: left = -(96.33) / (2 + 1.0) = -32.111, right = -(-96.33) / (4 + 1.0) = 19.267.
6. Prediction at bmi = 27.0 after one tree: 134.167 + 0.3 x 19.267 = **139.95**. More trees repeat steps 2 to 6 on the new errors.

## 6. K-Nearest Neighbors Regression (KNN)

**Working.** There is no training step. To predict, find the k rows closest to the query and average their y values. Features must be scaled first, otherwise the feature with the larger numbers dominates the distance.

**Equations.**

```
scaled value = (x - mean) / std
d = sqrt( (x1 - q1)^2 + (x2 - q2)^2 )     on scaled values
y_hat = mean of y for the k nearest rows
```

**Steps (k = 3, features bmi and bp).** bmi: mean 25.850, std 4.036. bp: mean 92.500, std 6.576.

| Row | bmi | bp | y | Scaled distance to query | Rank |
|---|---|---|---|---|---|
| 1 | 32.1 | 101 | 151 | 2.096 | 6 |
| 2 | 21.6 | 87 | 75 | 1.414 | 4 |
| 3 | 30.5 | 93 | 141 | 0.980 | 1 |
| 4 | 25.3 | 84 | 206 | 1.005 | 2 |
| 5 | 23.0 | 101 | 135 | 1.944 | 5 |
| 6 | 22.6 | 89 | 97 | 1.101 | 3 |

Nearest 3 rows: 3, 4, 6. y_hat = (141 + 206 + 97) / 3 = **148.00**

## Summary of the hand predictions

| Algorithm | Prediction at bmi = 27.0, bp = 90 |
|---|---|
| Linear Regression | 139.67 |
| Decision Tree (1 split) | 158.25 |
| Random Forest (3 trees) | 158.74 |
| XGBoost (1 tree) | 139.95 |
| KNN (k = 3) | 148.00 |

The neural network example shows one training step on scaled data, so it is not in this table. These are tiny teaching examples on six rows. The real comparison on the full dataset is in `regression_comparison.py`.
