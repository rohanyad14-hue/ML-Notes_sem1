# MACSE504 Machine Learning: FORMULA SHEET

<p class="cover"><b>Layout of every entry:</b> &nbsp; <b>Name | Function</b> &nbsp;→&nbsp; <b>Data on variables</b> (what each symbol is and what it does) &nbsp;→&nbsp; <b>Use case</b> (when to use it, with the professor's example numbers).<br>Sections <b>A-O</b> come from <b>Prof Provided Notes</b>; sections <b>P-T (Part B)</b> are standard forms for syllabus topics that folder does not cover. Items marked <i>(standard form)</i> are shown in the slides only as pictures or words, so the usual textbook form is given.</p>

**Sections:** A Regression · B Logistic Regression · C KNN · D Decision Trees · E Naïve Bayes · F SVM/SVR · G Classification Metrics · H Imbalanced Data · I Clustering · J Association Rules · K Semi-Supervised & Graph · L Reinforcement Learning · M PCA · N LLE · O t-SNE · **Part B:** P Ensembles · Q Polynomial/Ridge/Lasso/Bias-Variance · R Policy-based RL & Planning · S PCA Extensions · T Evaluation Extras, CV & Tuning

---

## A. Regression

### A1. Simple linear regression model
| Name | Function |
|---|---|
| Regression equation | $\displaystyle Y=\beta_0+\beta_1X+\varepsilon\qquad\hat Y=\beta_0+\beta_1X$ |

| Variable | What it is / what it does |
|---|---|
| $Y$ | **Dependent / target** variable: the outcome to predict (e.g. pizza sales in \$1000s) |
| $X$ | **Independent / predictor**: the input (e.g. students in 1000s) |
| $\beta_0$ | **Intercept**: value of $Y$ when $X=0$ (baseline) |
| $\beta_1$ | **Slope / weight**: change in $Y$ per 1-unit change in $X$; sign gives direction of relationship (negative = inverse) |
| $\varepsilon$ | **Residual / random error**; OLS chooses $\beta$'s to minimise $\sum\varepsilon^2$ |

**Use case:** predict a continuous value from one feature. Exercise: $\hat Y=15+0.12X$, $X=500$ kWh ⇒ \$75.

### A2. Slope and intercept (OLS)
| Name | Function |
|---|---|
| Slope $\hat\beta_1$ (definition) | $\displaystyle\hat\beta_1=\frac{\sum_{i=1}^n(X_i-\bar X)(Y_i-\bar Y)}{\sum_{i=1}^n(X_i-\bar X)^2}=\frac{\text{Cov}(X,Y)}{\text{Var}(X)}$ |
| Slope (calculation form) | $\displaystyle\hat\beta_1=\frac{\sum X_iY_i-\frac{(\sum X_i)(\sum Y_i)}{n}}{\sum X_i^2-\frac{(\sum X_i)^2}{n}}$ |
| Intercept $\hat\beta_0$ | $\displaystyle\hat\beta_0=\bar Y-\hat\beta_1\bar X$ |

| Variable | What it is / what it does |
|---|---|
| $n$ | number of data points |
| $\bar X,\ \bar Y$ | sample means of $X$ and $Y$; the fitted line always passes through $(\bar X,\bar Y)$ |
| $\sum X_iY_i,\ \sum X_i^2$ | column sums from the working table ($X,\ Y,\ X^2,\ Y^2,\ XY$) |

**Use case:** hand calculation. Estriol data: $\sum X=15,\sum Y=10,\sum X^2=55,\sum XY=37$ ⇒ $\hat\beta_1=0.70$, $\hat\beta_0=2-0.7(3)=-0.10$.

### A3. Multiple linear regression (hand method, 2 predictors)
| Name | Function |
|---|---|
| Model | $\displaystyle\hat y=b_0+b_1x_1+b_2x_2$ |
| Regression sums | $\displaystyle\sum x_a x_b=\sum X_aX_b-\frac{(\sum X_a)(\sum X_b)}{n}$ (for $a,b\in\{1,2,y\}$) |
| Coefficient $b_1$ | $\displaystyle b_1=\frac{(\sum x_2^2)(\sum x_1y)-(\sum x_1x_2)(\sum x_2y)}{(\sum x_1^2)(\sum x_2^2)-(\sum x_1x_2)^2}$ |
| Coefficient $b_2$ | $\displaystyle b_2=\frac{(\sum x_1^2)(\sum x_2y)-(\sum x_1x_2)(\sum x_1y)}{(\sum x_1^2)(\sum x_2^2)-(\sum x_1x_2)^2}$ |
| Intercept $b_0$ | $\displaystyle b_0=\bar y-b_1\bar x_1-b_2\bar x_2$ |

| Variable | What it is / what it does |
|---|---|
| $x_1,x_2$ | predictor variables (lower-case sums above are the *corrected* sums) |
| $b_1$ | change in $y$ per unit $x_1$ **holding $x_2$ fixed** |
| $b_2$ | change in $y$ per unit $x_2$ **holding $x_1$ fixed** |
| $b_0$ | mean of $y$ when all predictors are 0 |
| denominator | same for $b_1,b_2$; becomes 0 if $x_1,x_2$ are perfectly correlated (collinearity) |

**Use case:** the Statology example gives $\sum x_1^2=263.875,\ \sum x_2^2=194.875,\ \sum x_1y=1162.5,\ \sum x_2y=-953.5,\ \sum x_1x_2=-200.375$ ⇒ $\hat y=-6.867+3.148x_1-1.656x_2$.

### A4. Regression error metrics
| Name | Function |
|---|---|
| MAE | $\displaystyle\text{MAE}=\frac1n\sum_{i=1}^n\lvert y_i-\hat y_i\rvert$ |
| MSE | $\displaystyle\text{MSE}=\frac1n\sum_{i=1}^n(y_i-\hat y_i)^2$ |
| RMSE | $\displaystyle\text{RMSE}=\sqrt{\text{MSE}}$ |
| $R^2$ (coefficient of determination) | $\displaystyle R^2=1-\frac{SS_{res}}{SS_{tot}}=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ |

| Variable | What it is / what it does |
|---|---|
| $y_i,\ \hat y_i$ | actual and predicted target for sample $i$ |
| $n$ | number of test samples |
| $SS_{res}$ | residual sum of squares (error left after the model) |
| $SS_{tot}$ | total sum of squares about the mean $\bar y$ (error of the "predict the mean" baseline) |
| $R^2$ | fraction of variance explained, 0 to 1 (0.85 ⇒ 85 % explained). High $R^2$ can mean overfitting. |

**Use case:** MAE for outlier-robust reporting; MSE/RMSE when large errors must be punished (RMSE keeps target units). Example actual $[2,3,9,6]$, predicted $[1,2,8,1]$: MAE 2, MSE 7, RMSE 2.65.

---

## B. Logistic Regression

### B1. Odds, log-odds, sigmoid
| Name | Function |
|---|---|
| Odds | $\displaystyle\text{odds}=\frac{p}{1-p}$ |
| Log-odds (logit) | $\displaystyle\ln\frac{p}{1-p}=z=w\cdot X+b=\beta_0+\beta_1x$ |
| Sigmoid / logistic function | $\displaystyle p(x)=\frac{1}{1+e^{-z}}=\frac{e^{w\cdot X+b}}{1+e^{w\cdot X+b}}$ |

| Variable | What it is / what it does |
|---|---|
| $p$ | predicted probability of Class 1, always in $(0,1)$ |
| $z$ | linear score; $z=0\Leftrightarrow p=0.5$ |
| $w,\ b$ (or $\beta_1,\beta_0$) | weights (coefficients) and intercept found by **maximum likelihood**; intercept = log-odds when all features are 0 |
| threshold 0.5 | $p\ge0.5\Rightarrow$ Class 1, else Class 0 |

**Use case:** binary classification (spam / not spam, 0 / 1). Multinomial (≥3 unordered) and ordinal (≥3 ordered) are extensions.

### B2. Log loss / cost function
| Name | Function |
|---|---|
| Per-example cost | $\displaystyle\text{Cost}(h_\theta(x),y)=\begin{cases}-\log h_\theta(x)&y=1\\-\log\big(1-h_\theta(x)\big)&y=0\end{cases}$ |
| Log loss (binary cross-entropy) | $\displaystyle\text{LogLoss}=-\frac1n\sum_{i=1}^n\Big[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)\Big]$ |
| Weighted (cost-sensitive) form | $\displaystyle\text{Loss}=-w_1\,y\log p-w_0\,(1-y)\log(1-p)$ |

| Variable | What it is / what it does |
|---|---|
| $y_i\in\{0,1\}$ | true label |
| $\hat y_i,\ p$ | predicted probability of class 1; a confident wrong prediction makes the cost → ∞ |
| $w_0,\ w_1$ | class weights, e.g. `{0:1, 1:20}` ⇒ a missed minority example costs 20× |

**Use case:** the training loss for logistic regression (squared loss is for linear regression). Use the weighted form for imbalanced data.

---

## C. K-Nearest Neighbours

### C1. Distance measures
| Name | Function |
|---|---|
| Euclidean | $\displaystyle d(x,y)=\sqrt{\sum_{i=1}^m(x_i-y_i)^2}$ |
| Manhattan (city-block) | $\displaystyle d(x,y)=\sum_{i=1}^m\lvert x_i-y_i\rvert$ |
| Minkowski ($l$-norm) | $\displaystyle d(x,y)=\Big(\sum_{i=1}^m\lvert x_i-y_i\rvert^{r}\Big)^{1/r}$ |
| Chebyshev | $\displaystyle d(x,y)=\max_i\lvert x_i-y_i\rvert$ |
| Cosine similarity *(standard form)* | $\displaystyle\cos\theta=\frac{x\cdot y}{\lVert x\rVert\,\lVert y\rVert}$ |

| Variable | What it is / what it does |
|---|---|
| $x,\ y$ | two records (feature vectors) |
| $m$ | number of features |
| $r$ | Minkowski order: $r=1$ → Manhattan, $r=2$ → Euclidean, $r\to\infty$ → Chebyshev |

**Use case:** Euclidean is the default for numeric data; Manhattan is robust to outliers (K-means exercise uses it); cosine for text/angle similarity; Jaccard for binary match data.

### C2. Standardisation (z-score)
| Name | Function |
|---|---|
| z-score | $\displaystyle z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}$ |

| Variable | What it is / what it does |
|---|---|
| $x_{ij}$ | value of sample $i$ for feature $j$ |
| $\mu_j,\ \sigma_j$ | mean and standard deviation of feature $j$ over all samples |
| $z_{ij}$ | rescaled value (mean 0, std 1) so large-range features do not dominate the distance |

**Use case:** always before KNN/K-means when features have different units (weight in kg vs protein in ng/dL).

### C3. Weighted KNN, k rule, VDM
| Name | Function |
|---|---|
| Weighted vote (classification) | $\displaystyle\hat f(q)=\arg\max_{v\in V}\sum_{i=1}^kw_i\,\delta\big(v,f(x_i)\big)$ |
| Weighted average (regression) | $\displaystyle\hat f(q)=\frac{\sum_{i=1}^kw_if(x_i)}{\sum_{i=1}^kw_i}$ |
| Rule of thumb for $k$ | $k\approx\sqrt N$ |
| Value Difference Metric (nominal data) | $\displaystyle vdm_a(x,y)=\sum_{c=1}^C\Big(\frac{N_{a,x,c}}{N_{a,x}}-\frac{N_{a,y,c}}{N_{a,y}}\Big)^2$ |

| Variable | What it is / what it does |
|---|---|
| $q$ | query point |
| $w_i$ | weight of neighbour $i$ (often larger for nearer points); $w_i=1$ ⇒ unweighted |
| $\delta(v,f(x_i))$ | 1 if neighbour's class equals $v$, else 0 |
| $N$ | number of training points |
| $N_{a,x},\ N_{a,x,c}$ | times attribute $a$ took value $x$, and times it did so with class $c$ |
| $C$ | number of output classes |

**Use case:** $k$ odd to avoid ties; small $k$ = noisy, large $k$ = blurred boundaries. VDM compares categorical values by how similarly they relate to the classes.

---

## D. Decision Trees

### D1. Entropy and information gain (ID3)
| Name | Function |
|---|---|
| Entropy / expected information | $\displaystyle\text{Info}(D)=-\sum_{i=1}^mp_i\log_2p_i,\quad p_i=\frac{\lvert C_{i,D}\rvert}{\lvert D\rvert}$ |
| Entropy after splitting on $A$ | $\displaystyle\text{Info}_A(D)=\sum_{j=1}^v\frac{\lvert D_j\rvert}{\lvert D\rvert}\,\text{Info}(D_j)$ |
| Information gain | $\displaystyle\text{Gain}(A)=\text{Info}(D)-\text{Info}_A(D)$ |

| Variable | What it is / what it does |
|---|---|
| $D$ | the data partition (training tuples with labels) |
| $m$ | number of classes; $p_i$ = proportion of class $C_i$ in $D$ |
| $A$ | candidate attribute; it splits $D$ into $v$ subsets $D_1..D_v$ |
| $\text{Info}(D)$ | impurity in bits; 0 = pure, max = evenly mixed |
| $\text{Gain}(A)$ | entropy removed by the split; **pick the largest** (biased to many-valued attributes) |

**Use case:** AllElectronics: $\text{Info}(D)=0.940$ bits; $\text{Gain}(age)=0.246$ (highest ⇒ root), student 0.151, credit 0.048, income 0.029.

### D2. Gain ratio (C4.5)
| Name | Function |
|---|---|
| Split information | $\displaystyle\text{SplitInfo}_A(D)=-\sum_{j=1}^v\frac{\lvert D_j\rvert}{\lvert D\rvert}\log_2\frac{\lvert D_j\rvert}{\lvert D\rvert}$ |
| Gain ratio | $\displaystyle\text{GainRatio}(A)=\frac{\text{Gain}(A)}{\text{SplitInfo}_A(D)}$ |

| Variable | What it is / what it does |
|---|---|
| $\text{SplitInfo}$ | entropy of the *partition sizes*; large when an attribute splits into many small parts |
| $\text{GainRatio}$ | gain penalised for many branches; choose the maximum (with the safeguard: gain ≥ average gain) |

**Use case:** prevents choosing ID-like attributes. For age: SplitInfo (5,4,5 of 14) $=1.577$, ratio $=0.156$.

### D3. Gini index (CART)
| Name | Function |
|---|---|
| Gini impurity | $\displaystyle\text{Gini}(D)=1-\sum_{i=1}^mp_i^2$ |
| Gini after a binary split | $\displaystyle\text{Gini}_A(D)=\frac{\lvert D_1\rvert}{\lvert D\rvert}\text{Gini}(D_1)+\frac{\lvert D_2\rvert}{\lvert D\rvert}\text{Gini}(D_2)$ |
| Impurity reduction | $\displaystyle\Delta\text{Gini}(A)=\text{Gini}(D)-\text{Gini}_A(D)$ |

| Variable | What it is / what it does |
|---|---|
| $p_i$ | proportion of class $i$ in $D$ |
| $D_1,D_2$ | the two parts of a **binary** split; a discrete attribute with $v$ values has $2^v-2$ candidate splits |

**Use case:** CART picks the split with the **smallest** $\text{Gini}_A$ (largest reduction). $\text{Gini}(D)$ for 9 yes/5 no $=0.459$.

### D4. Pruning and cost
| Name | Function |
|---|---|
| Cost-complexity (CART) | $\displaystyle R_\alpha(T)=R(T)+\alpha\,\lvert N_L\rvert$ |
| Pessimistic error (C4.5) | $\displaystyle e_{adj}=e_{train}+\text{penalty}$ |
| MDL | $\displaystyle\min\big(\text{Bits}_{tree}+\text{Bits}_{exceptions}\big)$ |
| Growing cost | $\displaystyle O\big(n\times\lvert D\rvert\times\log\lvert D\rvert\big)$ |

| Variable | What it is / what it does |
|---|---|
| $R(T)$ | misclassification rate of tree $T$ |
| $\lvert N_L\rvert$ | number of leaves (tree complexity) |
| $\alpha$ | complexity penalty; bigger $\alpha$ ⇒ smaller tree |
| $n$ | number of attributes; $\lvert D\rvert$ = number of training tuples |

**Use case:** post-pruning to fight overfitting; compare a subtree's cost with the cost of replacing it by a leaf.

---

## E. Naïve Bayes

### E1. Bayes rule, MAP and the naïve classifier
| Name | Function |
|---|---|
| Bayes rule | $\displaystyle P(c\mid x)=\frac{P(x\mid c)\,P(c)}{P(x)}$ (Posterior $=\dfrac{\text{Likelihood}\times\text{Prior}}{\text{Evidence}}$) |
| Conditional / joint | $\displaystyle P(A\mid B)=\frac{P(A,B)}{P(B)},\qquad P(A,B)=P(A)P(B)\ \text{if independent}$ |
| Naïve Bayes (MAP) | $\displaystyle c^*=\arg\max_{c}\ P(c)\prod_{j=1}^nP(x_j\mid c)$ |

| Variable | What it is / what it does |
|---|---|
| $c,\ c^*$ | a class label and the predicted (maximum a posteriori) class |
| $x=(x_1..x_n)$ | feature vector of the instance to classify |
| $P(c)$ | **prior**: class frequency in the training set (9/14, 5/14) |
| $P(x_j\mid c)$ | **likelihood** of feature value given class, from a lookup table |
| $P(x)$ | evidence; same for all classes so it is dropped when comparing |

**Use case:** Play-Tennis: $P(\text{Yes}\mid x')\propto0.0053<P(\text{No}\mid x')\propto0.0206$ ⇒ "No". Spam filtering/text classification.

### E2. Gaussian likelihood and m-estimate
| Name | Function |
|---|---|
| Gaussian likelihood (continuous feature) | $\displaystyle\hat P(x_j\mid c_i)=\frac{1}{\sqrt{2\pi}\,\sigma_{ji}}\exp\!\Big(-\frac{(x_j-\mu_{ji})^2}{2\sigma_{ji}^2}\Big)$ |
| m-estimate (fixes zero probabilities) | $\displaystyle\hat P(a_{jk}\mid c_i)=\frac{n_c+mp}{n+m}$ |

| Variable | What it is / what it does |
|---|---|
| $\mu_{ji},\ \sigma_{ji}$ | mean and standard deviation of feature $j$ among training samples of class $c_i$ |
| $n$ | number of training examples of class $c_i$ |
| $n_c$ | of those, how many have feature value $a_{jk}$ |
| $p$ | prior estimate, usually $1/t$ for $t$ possible values |
| $m$ | number of "virtual" examples (≥ 1, up to ~1 % of data) |

**Use case:** temperature: $\mu_{Yes}=21.64,\sigma_{Yes}=2.35$; $\mu_{No}=23.88,\sigma_{No}=7.09$. $P(\text{overcast}\mid no)=\frac{0+1\cdot\frac13}{5+1}=\frac1{18}$.

---

## F. Support Vector Machines and SVR

### F1. Hard-margin SVM
| Name | Function |
|---|---|
| Hyperplane / decision rule | $\displaystyle w^Tx+b=0,\qquad f(x)=\operatorname{sign}(w^Tx+b)$ |
| Margin constraint | $\displaystyle y_i(w^Tx_i+b)\ge1\quad\forall i$ |
| Primal problem | $\displaystyle\min_{w,b}\ \tfrac12\lVert w\rVert^2\ \ \text{s.t.}\ \ y_i(w^Tx_i+b)\ge1$ |
| Margin width | $\displaystyle m=\frac{2}{\lVert w\rVert}$ |

| Variable | What it is / what it does |
|---|---|
| $w$ | normal vector to the hyperplane; its length sets the margin |
| $b$ | bias / offset of the hyperplane |
| $x_i,\ y_i\in\{+1,-1\}$ | training point and its class label |
| $\lVert w\rVert^2=w^Tw$ | minimising it **maximises the margin** |

**Use case:** linearly separable data; prefer the maximum-margin separator over any other boundary.

### F2. Dual problem and recovery of $w,b$
| Name | Function |
|---|---|
| Lagrangian multipliers | $\alpha_i\ge0$ (one per constraint) |
| Stationarity conditions | $\displaystyle w=\sum_i\alpha_iy_ix_i,\qquad\sum_i\alpha_iy_i=0$ |
| Dual (maximise) | $\displaystyle W(\alpha)=\sum_{i=1}^n\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,x_i^Tx_j,\quad\alpha_i\ge0,\ \sum\alpha_iy_i=0$ |
| Bias from two SVs | $\displaystyle b=-\tfrac12\big(w^Tx^{(+1)}+w^Tx^{(-1)}\big)$ |
| Classify new $z$ | $\displaystyle f(z)=\operatorname{sign}\Big(\sum_{j\in SV}\alpha_{t_j}y_{t_j}\,x_{t_j}^Tz+b\Big)$ |

| Variable | What it is / what it does |
|---|---|
| $\alpha_i$ | multiplier; **nonzero only for support vectors** (sparse solution) |
| $x_i^Tx_j$ | inner product (similarity); the only place the data enter, which makes the kernel trick possible |
| $t_j$ | indices of the $s$ support vectors |

**Use case:** train by quadratic programming (global optimum). Worked linear example: SVs $(1,0),(3,1),(3,-1)$ ⇒ $\alpha=(-3.5,0.75,0.75)$, $w=(1,0)$, $b=-2$.

### F3. Soft-margin SVM
| Name | Function |
|---|---|
| Soft constraints | $\displaystyle y_i(w^Tx_i+b)\ge1-\xi_i,\quad\xi_i\ge0$ |
| Primal | $\displaystyle\min\ \tfrac12\lVert w\rVert^2+C\sum_{i=1}^n\xi_i$ |
| Dual | same $W(\alpha)$ as F2 but with $\displaystyle0\le\alpha_i\le C,\ \sum\alpha_iy_i=0$ |

| Variable | What it is / what it does |
|---|---|
| $\xi_i$ | **slack**: how far point $i$ violates the margin; 0 = no error |
| $C$ | trade-off; **large $C$** ⇒ few errors (→ hard margin as $C\to\infty$); **small $C$** ⇒ wide, tolerant margin |
| $\mu_i$ | second multiplier; $C=\alpha_i+\mu_i$ gives the upper bound $C$ on $\alpha_i$ |

**Use case:** noisy or overlapping classes; soft margin is more robust to outliers.

### F4. Kernels
| Name | Function |
|---|---|
| Kernel (inner product in feature space) | $\displaystyle K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$ |
| Polynomial | $\displaystyle K(x,y)=(x^Ty+1)^d\quad(\text{or }(u\cdot v)^d)$ |
| RBF / Gaussian | $\displaystyle K(x,y)=\exp\!\Big(-\frac{\lVert x-y\rVert^2}{2\sigma^2}\Big)$ |
| Sigmoid | $\displaystyle K(x,y)=\tanh(\kappa\,x^Ty+\theta)$ |
| Kernel dual / classifier | replace every $x_i^Tx_j$ by $K(x_i,x_j)$: $\ f(z)=\sum_{SV}\alpha_iy_iK(x_i,z)+b$ |
| Building new kernels | $c_1k_1+c_2k_2,\ \ e^{k_1},\ \ k_1k_2,\ \ Q[k_1]\ (Q\text{ has non-negative coeffs}),\ \ f(x)k_1(x,y)f(y)$ |

| Variable | What it is / what it does |
|---|---|
| $\phi(\cdot)$ | mapping from input space to (higher-dimensional) feature space; never computed explicitly |
| $d$ | polynomial degree |
| $\sigma$ | RBF width: small ⇒ wiggly boundary, large ⇒ smooth; feature space is infinite-dimensional |
| $\kappa,\theta$ | sigmoid-kernel parameters (does not satisfy Mercer for all values) |
| Mercer's condition | a valid kernel's matrix is positive semi-definite |

**Use case:** non-linearly separable data (XOR). Example: $K(x,y)=(xy+1)^2,\ C=100$ ⇒ $f(z)=0.6667z^2-5.333z+9$. Start with low-degree polynomial or RBF.

### F5. Support Vector Regression
| Name | Function |
|---|---|
| Model | $f(x)=wx+b$ |
| Objective | $\displaystyle\min\ \tfrac12w^2+C\sum_{i=1}^n(\xi_i+\xi_i^*)$ |
| Constraints | $\displaystyle y_i-(wx_i+b)\le\varepsilon+\xi_i,\quad(wx_i+b)-y_i\le\varepsilon+\xi_i^*,\quad\xi_i,\xi_i^*\ge0$ |

| Variable | What it is / what it does |
|---|---|
| $\varepsilon$ | half-width of the insensitive tube; errors $\le\varepsilon$ cost nothing |
| $\xi_i,\ \xi_i^*$ | slack for points above / below the tube |
| $C$ | penalty weight for points outside the tube |
| $w$ small | flat function (regularisation) |

**Use case:** $\varepsilon=0.5,\ C=10$, line $1.5x+0.5$ through (1,2),(2,3),(3,5): all slacks 0, cost $\tfrac12(1.5)^2=1.125$.

---

## G. Classification Metrics

### G1. Confusion-matrix metrics
| Name | Function |
|---|---|
| Accuracy | $\displaystyle\frac{TP+TN}{TP+TN+FP+FN}$ |
| Precision (PPV) | $\displaystyle\frac{TP}{TP+FP}$ |
| Recall / Sensitivity / TPR | $\displaystyle\frac{TP}{TP+FN}$ |
| Specificity (TNR) | $\displaystyle\frac{TN}{TN+FP}$ |
| False-positive rate | $\displaystyle FPR=\frac{FP}{FP+TN}=1-\text{Specificity}$ |
| NPV | $\displaystyle\frac{TN}{TN+FN}$ |
| F1 score | $\displaystyle F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}$ |
| $F_\beta$ (weighted harmonic mean) | $\displaystyle F=\frac{(\beta^2+1)PR}{\beta^2P+R}=\frac{1}{\alpha\frac1P+(1-\alpha)\frac1R}$ |

| Variable | What it is / what it does |
|---|---|
| TP / TN | correct positive / correct negative predictions |
| FP (Type I) | predicted positive, actually negative |
| FN (Type II) | predicted negative, actually positive |
| $P,\ R$ | precision and recall |
| $\beta,\ \alpha$ | $\beta>1$ favours recall; $\alpha=0.5$ ($\beta=1$) balances them; harmonic mean stays low unless both are high |

**Use case:** accuracy only on balanced data; for rare-event problems use precision/recall/F1. Cat–dog: TP 6, TN 11, FP 2, FN 1 ⇒ acc 0.85, P 0.75, R 0.857, F1 0.80.

### G2. Multi-class averaging and ROC
| Name | Function |
|---|---|
| Macro average | $\displaystyle\frac1K\sum_{k=1}^K\text{metric}_k$ |
| Micro average (precision) | $\displaystyle\frac{\sum_kTP_k}{\sum_kTP_k+\sum_kFP_k}$ |
| Weighted average | $\displaystyle\sum_k\frac{\text{support}_k}{N}\text{metric}_k$ |
| ROC axes | $\text{TPR}=\dfrac{TP}{TP+FN}$ vs $\text{FPR}=\dfrac{FP}{FP+TN}$ over all thresholds |

| Variable | What it is / what it does |
|---|---|
| $K$ | number of classes |
| support$_k$ | number of true samples of class $k$ |
| AUC | area under ROC; 1 = perfect, 0.5 = random diagonal |

**Use case:** macro treats every class equally (rare classes count), micro is dominated by large classes (urgent/normal/spam: macro .60, micro .73), weighted reflects class sizes.

---

## H. Imbalanced Data

### H1. Class weights and SMOTE
| Name | Function |
|---|---|
| Class weight | $\displaystyle w_{pos}=\frac{\%\text{ majority}}{\%\text{ minority}}\ \ (=99.997/0.003\approx33332),\quad w_{neg}=1$ |
| SMOTE synthetic point *(standard form)* | $\displaystyle x_{new}=x_i+\lambda\,(x_{nn}-x_i),\quad\lambda\sim U(0,1)$ |
| Resampling sizes | over-sampling: $\lvert S'_{min}\rvert=\lvert S_{min}\rvert+\lvert E\rvert$; under-sampling: $\lvert S'_{maj}\rvert=\lvert S_{maj}\rvert-\lvert E\rvert$ |

| Variable | What it is / what it does |
|---|---|
| $x_i$ | a minority sample |
| $x_{nn}$ | one of its $k$ nearest *minority* neighbours (j of them chosen at random) |
| $\lambda$ | random fraction placing the new point somewhere on the line between them |
| $E$ | set of generated (or removed) samples |

**Use case:** Abalone AUC (logistic regression): 0.684 baseline → 0.839 random over-sampling → 0.791 SMOTE → 0.647 under-sampling.

---

## I. Clustering

### I1. K-means
| Name | Function |
|---|---|
| Assignment step | $\displaystyle C(x)=\arg\min_{k}\ d(x,\mu_k)$ |
| Centroid update | $\displaystyle\mu_k=\frac{1}{\lvert C_k\rvert}\sum_{x\in C_k}x$ |
| Objective | $\displaystyle\min\sum_{k=1}^K\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$ |
| Complexity | $O(tKn)$ |

| Variable | What it is / what it does |
|---|---|
| $K$ | number of clusters (chosen in advance) |
| $\mu_k$ | centroid (mean) of cluster $C_k$ |
| $d$ | distance (Euclidean, or Manhattan in the exercise) |
| $t,\ n$ | iterations, objects ($K,t\ll n$) |

**Use case:** A(1,1),B(2,1),C(4,3),D(5,4), $K=2$ ⇒ clusters {A,B},{C,D}, centroids (1.5,1) and (4.5,3.5).

### I2. Hierarchical-clustering distances
| Name | Function |
|---|---|
| Minimum (single link) | $\displaystyle d_{min}(C_i,C_j)=\min_{p\in C_i,\,p'\in C_j}\lvert p-p'\rvert$ |
| Maximum (complete link) | $\displaystyle d_{max}(C_i,C_j)=\max_{p\in C_i,\,p'\in C_j}\lvert p-p'\rvert$ |
| Mean | $\displaystyle d_{mean}(C_i,C_j)=\lvert m_i-m_j\rvert$ |
| Average | $\displaystyle d_{avg}(C_i,C_j)=\frac{1}{n_in_j}\sum_{p\in C_i}\sum_{p'\in C_j}\lvert p-p'\rvert$ |
| Update after merging $c_i,c_j$ | single: $\min(d(c_i,c_k),d(c_j,c_k))$; complete: $\max(d(c_i,c_k),d(c_j,c_k))$ |

| Variable | What it is / what it does |
|---|---|
| $\lvert p-p'\rvert$ | distance between two objects |
| $m_i,\ n_i$ | mean and size of cluster $C_i$ |

**Use case:** single link ⇒ chaining/long clusters; complete link ⇒ tight spherical clusters. Italian-city example single-link heights 138, 219, 255, 268, 295.

### I3. WCSS and silhouette
| Name | Function |
|---|---|
| Within-cluster SS | $\displaystyle W(C_k)=\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2,\qquad\text{WCSS}=\sum_{k=1}^KW(C_k)$ |
| Cohesion $a(i)$ | $\displaystyle a(i)=\frac{1}{\lvert C_I\rvert-1}\sum_{j\in C_I,\,j\neq i}d(i,j)$ |
| Separation $b(i)$ | $\displaystyle b(i)=\min_{J\neq I}\frac{1}{\lvert C_J\rvert}\sum_{j\in C_J}d(i,j)$ |
| Silhouette | $\displaystyle s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}},\ \ s(i)=0\text{ if }\lvert C_I\rvert=1$ |

| Variable | What it is / what it does |
|---|---|
| $C_I$ | cluster containing point $i$; $C_J$ any other cluster |
| $a(i)$ | mean distance to own cluster (want small) |
| $b(i)$ | mean distance to nearest other cluster (want large) |
| $s(i)$ | in $[-1,1]$: +1 well clustered, 0 on boundary, −1 probably misassigned |

**Use case:** WCSS vs $K$ for the elbow method; silhouette averaged over points to choose $K$ (no labels needed).

### I4. Davies-Bouldin, Calinski-Harabasz, Rand/ARI
| Name | Function |
|---|---|
| Davies-Bouldin | $\displaystyle DB=\frac1k\sum_{i=1}^kR_i,\qquad R_i=\max_{j\neq i}\frac{s_i+s_j}{d_{ij}}$ |
| Calinski-Harabasz | $\displaystyle CH=\frac{\text{Between-cluster dispersion}}{\text{Within-cluster dispersion}}$ (variance-ratio criterion) |
| Rand index | $\displaystyle RI=\frac{a+d}{a+b+c+d}$ |
| Adjusted Rand index | $\displaystyle ARI=\frac{RI-E[RI]}{1-E[RI]}$ |

| Variable | What it is / what it does |
|---|---|
| $k$ | number of clusters |
| $s_i$ | scatter of cluster $i$ (mean distance of its points to its centroid) |
| $d_{ij}$ | distance between centroids of clusters $i$ and $j$ |
| $a,d$ | pairs placed together / apart in **both** partitions (agreement) |
| $b,c$ | pairs on which the two partitions disagree |
| $E[RI]$ | expected RI under random labelling |

**Use case:** DB: lower is better (0 optimal). CH: higher is better (compact spherical clusters). ARI: needs true labels; 1 = perfect, ≈0 = random.

---

## J. Association Rules

### J1. Support, confidence, Apriori
| Name | Function |
|---|---|
| Support of $A\Rightarrow B$ | $\displaystyle\text{supp}(A\Rightarrow B)=P(A\cup B)=\frac{\#\text{transactions containing }A\cup B}{\#\text{transactions}}$ |
| Confidence | $\displaystyle\text{conf}(A\Rightarrow B)=P(B\mid A)=\frac{P(A\cup B)}{P(A)}$ |
| Apriori candidate generation | join: $C_k=L_{k-1}\bowtie L_{k-1}$ (same first $k-2$ items); prune: delete $c$ if any $(k-1)$-subset $\notin L_{k-1}$ |
| Apriori property | every subset of a frequent itemset is frequent |

| Variable | What it is / what it does |
|---|---|
| $A,\ B$ | itemsets (antecedent, consequent) |
| $L_k$ | frequent $k$-itemsets (support ≥ minimum) |
| $C_k$ | candidate $k$-itemsets |
| min support / min confidence | user thresholds; rules passing both are **strong** |

**Use case:** basket analysis, e.g. $A\Rightarrow C$ (50 %, 66.6 %). FP-Growth builds an FP-tree and mines conditional pattern bases without candidate generation (e.g. $m$-conditional base {fca:2, fcab:1}).

---

## K. Semi-Supervised and Graph Learning

### K1. Label Propagation
| Name | Function |
|---|---|
| RBF similarity | $\displaystyle W_{ij}=\exp\!\Big(-\frac{\lVert x_i-x_j\rVert^2}{2\sigma^2}\Big),\quad W_{ii}=0$ |
| Transition matrix | $\displaystyle P=D^{-1}W,\quad D_{ii}=\sum_jW_{ij}$ |
| Iteration + hard clamp | $\displaystyle Y^{(t+1)}=PY^{(t)},\ \text{then}\ Y^{(t+1)}_{labelled}=Y^{(0)}_{labelled}$ |
| Final label | $\displaystyle\hat y_i=\arg\max_cY_{ic}$ |

| Variable | What it is / what it does |
|---|---|
| $W$ | symmetric similarity (edge-weight) matrix ($N\times N$) |
| $\sigma$ | RBF width; controls how fast similarity decays with distance |
| $D$ | diagonal degree matrix (row sums of $W$) |
| $Y$ | $N\times C$ soft-label matrix ($N$ nodes, $C$ classes); labelled rows one-hot, unlabelled rows 0 |

**Use case:** few labels + similarity graph; labels flow from sources to neighbours. Worked example converges to node 3 $=[0.368,0.632]$, node 4 $=[0.158,0.842]$.

### K2. Label Spreading and GNN step
| Name | Function |
|---|---|
| Normalised similarity | $\displaystyle S=D^{-1/2}WD^{-1/2}$ |
| Label Spreading update | $\displaystyle F^{(t+1)}=\alpha\,S\,F^{(t)}+(1-\alpha)\,Y,\qquad F^{(0)}=Y$ |
| GNN mean aggregation | $\displaystyle h_{N(i)}=\frac{1}{\lvert N(i)\rvert}\sum_{j\in N(i)}h_j^{(0)}$ |
| GNN layer | $\displaystyle z_i=W\,h_{N(i)},\qquad h_i^{(1)}=\text{ReLU}(z_i)=\max(0,z_i)$ |

| Variable | What it is / what it does |
|---|---|
| $\alpha\in(0,1)$ | **clamping parameter**: share taken from neighbours; $(1-\alpha)$ = share of the node's original label kept (soft clamp). Notes use 0.5 in the example, 0.2 typical. |
| $F^{(t)}$ | current soft-label matrix at iteration $t$ |
| $N(i)$ | neighbours of $i$ **plus itself** (self-loop) |
| $W$ (GNN) | shared learnable layer weight matrix |
| ReLU | non-linearity applied after the linear map |

**Use case:** noisy labels, graph/cluster data (images). GNN example: $h_A=[1,0],h_B=[0,2],h_C=[3,1]$ ⇒ mean $[1.33,1.00]$ ⇒ after the linear map $z=[3.67,2.00]$ (ReLU leaves it unchanged).

---

## L. Reinforcement Learning

### L1. Return, value, Bellman equations
| Name | Function |
|---|---|
| Finite-horizon value | $\displaystyle V^\pi(s_t)=E\big[r_{t+1}+\dots+r_{t+T}\big]=E\Big[\sum_{i=1}^Tr_{t+i}\Big]$ |
| Infinite-horizon (discounted) value | $\displaystyle V^\pi(s_t)=E\Big[\sum_{i=1}^\infty\gamma^{i-1}r_{t+i}\Big]$ |
| Optimal value | $\displaystyle V^*(s_t)=\max_\pi V^\pi(s_t)=\max_{a_t}Q^*(s_t,a_t)$ |
| Bellman equation for $V^*$ | $\displaystyle V^*(s_t)=\max_{a_t}\Big(E[r_{t+1}]+\gamma\sum_{s_{t+1}}P(s_{t+1}\mid s_t,a_t)V^*(s_{t+1})\Big)$ |
| Bellman equation for $Q^*$ | $\displaystyle Q^*(s_t,a_t)=E[r_{t+1}]+\gamma\sum_{s_{t+1}}P(s_{t+1}\mid s_t,a_t)\max_{a_{t+1}}Q^*(s_{t+1},a_{t+1})$ |
| Optimal policy | $\displaystyle\pi^*(s)=\arg\max_aQ^*(s,a)$ |
| One-step $Q$ (deterministic, used in exercises) | $\displaystyle Q(S,a)=R+\gamma\,V(S')$ |

| Variable | What it is / what it does |
|---|---|
| $s_t,\ a_t$ | state and action at time $t$ |
| $r_{t+1}$ | reward received after acting |
| $\pi$ | **policy**: mapping state → action |
| $V^\pi(s)$ | expected cumulative reward from $s$ following $\pi$ |
| $Q(s,a)$ | value of taking $a$ in $s$ then following the optimal policy |
| $\gamma\in[0,1)$ | **discount rate**: 0 = only immediate reward, near 1 = far-sighted; keeps the sum finite |
| $P(s'\mid s,a)$ | transition probability (the environment "model") |
| $T$ | horizon length (finite-horizon case) |

**Use case:** known-model planning. Example: Leave gives $Q=10+0.5(0)=10$; Stay gives $2+0.5\cdot10=7$ under the optimum ⇒ $\pi^*=$ Leave.

### L1b. Value functions and how they relate (Value-Based RL)
| Name | Function |
|---|---|
| State-value | $\displaystyle V^\pi(s)=E_\pi\Big[\sum_{k\ge0}\gamma^kr_{t+k+1}\ \Big|\ s_t=s\Big]$ |
| Action-value | $\displaystyle Q^\pi(s,a)=E_\pi\Big[\sum_{k\ge0}\gamma^kr_{t+k+1}\ \Big|\ s_t=s,\ a_t=a\Big]$ |
| $V$ from $Q$ | $\displaystyle V^\pi(s)=\sum_a\pi(a\mid s)\,Q^\pi(s,a),\qquad V^*(s)=\max_aQ^*(s,a)$ |
| $Q$ from $V$ | $\displaystyle Q^\pi(s,a)=E[r]+\gamma\sum_{s'}P(s'\mid s,a)\,V^\pi(s')$ |
| Bellman expectation eq. | $\displaystyle V^\pi(s)=\sum_a\pi(a\mid s)\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\Big]$ |
| Advantage | $\displaystyle A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ |

| Variable | What it is / what it does |
|---|---|
| $V^\pi(s)$ | "how good is it to **be** in $s$" if we follow policy $\pi$ |
| $Q^\pi(s,a)$ | "how good is it to **do $a$** in $s$ and then follow $\pi$" |
| $\pi(a\mid s)$ | policy: probability of choosing $a$ in $s$ |
| $R(s,a)$ | expected immediate reward |
| $A$ | positive ⇒ action better than the state's average |

**Use case:** the same numbers appear in value iteration (N18i): $V^*(S_2)=20$, $V^*(S_1)=18$.

### L2. Model-based: value iteration and policy iteration
| Name | Function |
|---|---|
| Value-iteration update | $\displaystyle Q(s,a)=E[r\mid s,a]+\gamma\sum_{s'}P(s'\mid s,a)V(s'),\qquad V(s)\leftarrow\max_aQ(s,a)$ |
| Convergence test | $\displaystyle\max_{s}\lvert V^{(l+1)}(s)-V^{(l)}(s)\rvert<\delta$ |
| Policy evaluation | $\displaystyle V^\pi(s)=E[r\mid s,\pi(s)]+\gamma\sum_{s'}P(s'\mid s,\pi(s))V^\pi(s')$ (linear equations) |
| Policy improvement | $\displaystyle\pi'(s)=\arg\max_a\Big(E[r\mid s,a]+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\Big)$ |
| Cost per iteration | value iter. $O(\lvert S\rvert^2\lvert A\rvert)$; policy iter. $O(\lvert A\rvert\lvert S\rvert^2+\lvert S\rvert^3)$ |

| Variable | What it is / what it does |
|---|---|
| $V(s)$ | current value estimate of state $s$ |
| $\delta$ | small tolerance that decides when to stop |
| $l$ | iteration counter |
| $\lvert S\rvert,\lvert A\rvert$ | number of states / actions |

**Use case:** when $P$ and the reward model are known (dynamic programming, no exploration). Policy iteration needs fewer iterations but each one is costlier.

### L3. Exploration and bandit update
| Name | Function |
|---|---|
| ε-greedy | with prob. $\varepsilon$ pick a random action, with prob. $1-\varepsilon$ pick $\arg\max_aQ(s,a)$ |
| Softmax with temperature | $\displaystyle P(a\mid s)=\frac{\exp[Q(s,a)/T]}{\sum_{b\in\mathcal A}\exp[Q(s,b)/T]}$ |
| Bandit / delta rule | $\displaystyle Q_{t+1}(a)\leftarrow Q_t(a)+\eta\,[\,r_{t+1}(a)-Q_t(a)\,]$ |

| Variable | What it is / what it does |
|---|---|
| $\varepsilon$ | exploration probability; start high, decrease over time |
| $T$ | temperature: **large ⇒ near-uniform (explore)**, **small ⇒ favour best (exploit)**; lowering $T$ = annealing |
| $\eta$ | learning rate; averages stochastic rewards, converges to the mean reward |

**Use case:** choose actions while learning (explore vs exploit); K-armed bandit lever values.

### L4. Q-Learning and Temporal-Difference methods
| Name | Function |
|---|---|
| TD(0) for $V$ | $\displaystyle V(s_t)\leftarrow V(s_t)+\eta\,[\,r_{t+1}+\gamma V(s_{t+1})-V(s_t)\,]$ |
| Q-learning, deterministic | $\displaystyle\hat Q(s_t,a_t)\leftarrow r_{t+1}+\gamma\max_{a'}\hat Q(s_{t+1},a')$ |
| Q-learning, general (**off-policy**) | $\displaystyle Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,R+\gamma\max_{a'}Q(s',a')-Q(s,a)\,]$ |
| Equivalent blended form | $\displaystyle Q(s,a)\leftarrow(1-\alpha)Q(s,a)+\alpha\,[\,R+\gamma\max_{a'}Q(s',a')\,]$ |
| SARSA (**on-policy**) | $\displaystyle Q(s,a)\leftarrow Q(s,a)+\eta\,[\,r+\gamma Q(s',a')-Q(s,a)\,]$ |

| Variable | What it is / what it does |
|---|---|
| $R\ (r)$ | immediate reward |
| $s',\ a'$ | next state, next action (in SARSA, $a'$ is the action actually chosen by the policy) |
| $\max_{a'}Q(s',a')$ | best value achievable from the next state |
| $R+\gamma\max Q(s',a')$ | **TD target** (backed-up estimate) |
| bracket $[\,\cdot\,]$ | **TD error**: target minus current estimate |
| $\alpha\ (\eta)$ | learning rate, $0\le\alpha\le1$; $\alpha=1$ overwrites the old value |

**Use case:** model-free learning from experience. Example: $Q=12,R=5,\max Q'=30,\alpha=0.2,\gamma=0.8$ ⇒ $0.8(12)+0.2(5+24)=15.4$. 4-state chain with $\alpha=1,\gamma=0.9$: $Q=100,\,90,\,81$ going backwards from the goal.

### L4b. Q-Learning algorithm card
| Name | Function |
|---|---|
| 1. Initialise | $\displaystyle Q(s,a)=0\quad\forall s,a\ \ (\text{terminal-state values stay }0)$ |
| 2. Choose action (ε-greedy) | $\displaystyle a=\begin{cases}\text{random action}&\text{prob. }\varepsilon\\\arg\max_aQ(s,a)&\text{prob. }1-\varepsilon\end{cases}$ |
| 3. Observe | reward $R$ and next state $s'$ after taking $a$ |
| 4. TD target | $\displaystyle y=R+\gamma\max_{a'}Q(s',a')\quad(\max=0\text{ if }s'\text{ terminal})$ |
| 5. TD error | $\displaystyle\delta=y-Q(s,a)$ |
| 6. Update one cell | $\displaystyle Q(s,a)\leftarrow Q(s,a)+\alpha\,\delta$ |
| 7. Final policy | $\displaystyle\pi^*(s)=\arg\max_aQ(s,a)$ |
| Convergence conditions | all $(s,a)$ visited infinitely often; $\displaystyle\sum_t\alpha_t=\infty,\ \ \sum_t\alpha_t^2<\infty$ |
| Value of a state far from the goal | $\displaystyle Q=R\,\gamma^{N-1}$ for a state $N$ steps from the goal (deterministic, $\alpha=1$) |

| Variable | What it is / what it does |
|---|---|
| $\alpha$ | learning rate: how far one update moves $Q$ toward the target |
| $\gamma$ | discount: weight of future reward (0.9 is typical in the notes) |
| $\varepsilon$ | exploration probability; decay it as learning progresses |
| $y$ | TD target = what $Q(s,a)$ *should* be according to this one sample |
| $\delta$ | TD error = surprise; 0 when $Q$ already matches the target |

**Use case:** every Q-learning exam question. Example (N18g, $\alpha=0.5,\gamma=0.9$): $Q(C,R)$ after two goal visits $=5\to7.5$; then $Q(A,D)=2.25+0.5\,[\,0+0.9(7.5)-2.25\,]=4.5$. SARSA differs only in step 4: use $Q(s',a')$ of the action actually taken instead of $\max$.

### L5. Eligibility traces and Sarsa($\lambda$)
| Name | Function |
|---|---|
| Eligibility trace | $\displaystyle e_t(s,a)=\begin{cases}1&\text{if }s=s_t\text{ and }a=a_t\\\gamma\lambda\,e_{t-1}(s,a)&\text{otherwise}\end{cases}$ |
| TD error (Sarsa) | $\displaystyle\delta_t=r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)$ |
| Sarsa($\lambda$) update | $\displaystyle Q(s,a)\leftarrow Q(s,a)+\eta\,\delta_t\,e_t(s,a)\quad\forall s,a$ |

| Variable | What it is / what it does |
|---|---|
| $e_t(s,a)$ | "how recently/often" pair $(s,a)$ was visited; starts 0 |
| $\lambda\in[0,1]$ | trace decay: $\lambda=0$ ⇒ one-step update only (= Sarsa(0)/TD(0)); $\lambda\to1$ ⇒ credit flows to all earlier steps, decaying only by $\gamma$ |
| $\delta_t$ | temporal-difference error at time $t$ |

**Use case:** faster credit assignment over long action sequences; every eligible pair is updated, not just the last one.

---

## M. Principal Component Analysis (PCA)

### M1. Covariance, eigen-decomposition, projection
| Name | Function |
|---|---|
| Mean-centring | $\displaystyle X_c=X-\bar X$ |
| Covariance | $\displaystyle\text{Cov}(A,B)=\frac{\sum(a_i-\bar a)(b_i-\bar b)}{N}\ \ (\text{population})\ \text{ or }\ \frac{\sum(a_i-\bar a)(b_i-\bar b)}{N-1}\ (\text{sample})$ |
| Covariance matrix | $\displaystyle S=\begin{bmatrix}\text{Var}(a)&\text{Cov}(a,b)\\\text{Cov}(b,a)&\text{Var}(b)\end{bmatrix}$ |
| Characteristic equation | $\displaystyle\det(S-\lambda I)=0$ |
| Eigenvector | $\displaystyle(S-\lambda I)\,v=0,\qquad u=\frac{v}{\lVert v\rVert}$ |
| Projection (PC score) | $\displaystyle z=X_c\cdot u_1=u_1^T(x-\bar x)$ |
| Reconstruction | $\displaystyle\hat X=z\,u_1^T+\bar X$ |
| Explained variance ratio | $\displaystyle\frac{\lambda_i}{\sum_k\lambda_k}$ |

| Variable | What it is / what it does |
|---|---|
| $S$ | $D\times D$ covariance of mean-centred features (captures linear dependence) |
| $\lambda$ | **eigenvalue**: variance of the data along that new axis |
| $I$ | identity matrix ($D\times D$) |
| $\det$ | determinant; setting it to 0 gives a degree-$D$ polynomial in $\lambda$ |
| $v,\ u$ | eigenvector and its unit-length version = **principal-component direction** (PC1 has the largest $\lambda$) |
| $z$ | score = coordinate of a point along the PC |
| $N$ | number of samples |

**Use case:** reduce dimension while keeping most variance; keep PCs until ≈ 80 % cumulative variance (scree plot). 5-point example: $S=\begin{bmatrix}2&1.6\\1.6&2\end{bmatrix}$, $\lambda=3.6,0.4$ (PC1 = 90 %). Hand example: $S=\begin{bmatrix}14&-11\\-11&23\end{bmatrix}$, $\lambda=30.38,\,6.62$; PC1 scores $-4.305,\,3.736,\,5.692,\,-5.123$.

---

## N. Locally Linear Embedding (LLE)

### N1. Weights and embedding
| Name | Function |
|---|---|
| Neighbour distance | $\displaystyle d(X_i,X_j)=\lVert X_i-X_j\rVert$ |
| Reconstruction error | $\displaystyle\varepsilon(W)=\sum_i\Big\lVert X_i-\sum_jW_{ij}X_j\Big\rVert^2,\quad W_{ij}=0\text{ for non-neighbours},\ \ \sum_jW_{ij}=1$ |
| Local Gram matrix | $\displaystyle C_{jk}=(X_i-X_j)^T(X_i-X_k)$ |
| Closed-form weights | $\displaystyle W_i=\frac{C^{-1}\mathbf 1}{\mathbf 1^TC^{-1}\mathbf 1}$ (solve $Cw'=\mathbf1$, then divide by $\sum w'$) |
| Embedding cost | $\displaystyle\Phi(Y)=\sum_i\Big\lVert Y_i-\sum_jW_{ij}Y_j\Big\rVert^2=\text{Tr}(YMY^T)$ |
| Cost matrix | $\displaystyle M=(I-W)^T(I-W)$ |

| Variable | What it is / what it does |
|---|---|
| $K$ | number of nearest neighbours (small ⇒ loses topology, large ⇒ loses local linearity) |
| $W_{ij}$ | how much neighbour $j$ contributes to rebuilding point $i$; rows sum to 1; diagonal is 0 |
| $C$ | local covariance (Gram) matrix of point $i$'s neighbourhood |
| $\mathbf1$ | vector of ones |
| $Y_i\in\mathbb R^d$ | low-dimensional coordinates |
| $M$ | sparse, symmetric $N\times N$; rows sum to 0 |
| constraints | $\sum_iY_i=0$ (centred), $\frac1NYY^T=I$ (unit covariance) |

**Use case:** non-linear data on a manifold (Swiss roll). Embedding $=$ eigenvectors of $M$ with the 2nd…(d+1)-th **smallest** eigenvalues; discard the first (eigenvalue 0, vector $[1,\dots,1]^T$). Example: $G=\text{diag}(1,1,4)\Rightarrow W=(0.444,0.444,0.111)$.

---

## O. t-SNE

### O1. Probabilities and cost
| Name | Function |
|---|---|
| High-D similarity *(standard form)* | $\displaystyle p_{j\mid i}=\frac{\exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\lVert x_i-x_k\rVert^2/2\sigma_i^2)},\quad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2N}$ |
| Low-D similarity (Student-t, 1 d.o.f.) *(standard form)* | $\displaystyle q_{ij}=\frac{(1+\lVert y_i-y_j\rVert^2)^{-1}}{\sum_{k\neq l}(1+\lVert y_k-y_l\rVert^2)^{-1}}$ |
| Cost (KL divergence) | $\displaystyle C=KL(P\,\Vert\,Q)=\sum_{i}\sum_{j}p_{ij}\log\frac{p_{ij}}{q_{ij}}$ |
| Perplexity *(standard form)* | $\displaystyle\text{Perp}(P_i)=2^{H(P_i)},\quad H(P_i)=-\sum_jp_{j\mid i}\log_2p_{j\mid i}$ |

| Variable | What it is / what it does |
|---|---|
| $x_i,\ y_i$ | point $i$ in the original space and in the 2-D/3-D map |
| $\sigma_i$ | Gaussian bandwidth for point $i$, tuned so the perplexity matches the user's setting |
| Perplexity | smooth "effective number of neighbours", typically **5-50** |
| $p_{ij}$ | high-D joint probability (target) |
| $q_{ij}$ | low-D joint probability; **heavy tails** of the t-distribution solve the crowding problem |
| $KL$ | mismatch between $P$ and $Q$; minimised by gradient descent; goal $C\to0$ |

**Use case:** visualising high-dimensional data as 2-D/3-D clusters. Not for feature extraction; PCA is for global linear variance, LLE for local non-linear geometry, t-SNE for visual cluster structure.

<div class="pb"></div>

# PART B: Formulas for Syllabus Topics Not Covered by the Professor's Files

> ⚠ **Source note:** sections P-T are **not from Prof Provided Notes**; they are standard forms for the remaining syllabus topics (random forest/boosting, polynomial and regularised regression, policy-based RL, PCA extensions, cross-validation and tuning). Matching theory is in Theory sections 20-26; worked numbers in Numericals N19-N29.

---

## P. Ensembles: Bagging, Random Forest, Boosting

### P1. Bootstrap, OOB and ensemble variance
| Name | Function |
|---|---|
| Prob. a row is left out of a bootstrap | $\displaystyle\Big(1-\frac1n\Big)^n\ \xrightarrow{n\to\infty}\ e^{-1}\approx0.368$ |
| Bagging prediction | classification: $\hat y=\text{mode}\{h_b(x)\}_{b=1}^B$; regression: $\displaystyle\hat y=\frac1B\sum_{b=1}^Bh_b(x)$ |
| Variance of the average of $B$ correlated trees | $\displaystyle\text{Var}=\rho\,\sigma^2+\frac{1-\rho}{B}\,\sigma^2$ |

| Variable | What it is / what it does |
|---|---|
| $n$ | training-set size; each bootstrap draws $n$ rows **with replacement** |
| $B$ | number of trees/models |
| $h_b$ | the $b$-th tree |
| $\sigma^2$ | variance of one tree's prediction |
| $\rho$ | correlation between trees; random feature subsets in Random Forest **lower $\rho$** |
| 0.368 / 0.632 | out-of-bag share / in-bag share of distinct rows; OOB rows give a free validation score |

**Use case:** $\sigma^2=4,\rho=0.5,B=10$ ⇒ variance $2.2$ (vs 4 for one tree). Random Forest also uses $m\approx\sqrt p$ features per split.

### P2. AdaBoost and gradient boosting
| Name | Function |
|---|---|
| Weighted error of learner $t$ | $\displaystyle\varepsilon_t=\sum_{i:\,h_t(x_i)\neq y_i}w_i$ |
| Learner weight | $\displaystyle\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ |
| Sample-weight update | $\displaystyle w_i\leftarrow\frac{w_i\,e^{-\alpha_t\,y_ih_t(x_i)}}{Z_t}$ |
| Final AdaBoost classifier | $\displaystyle H(x)=\operatorname{sign}\Big(\sum_{t=1}^T\alpha_t\,h_t(x)\Big)$ |
| Gradient boosting (squared loss) | $\displaystyle F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\quad h_m\ \text{fit to}\ r_i=y_i-F_{m-1}(x_i)$ |

| Variable | What it is / what it does |
|---|---|
| $w_i$ | weight of training example $i$ (start at $1/n$) |
| $y_i,\ h_t(x_i)\in\{\pm1\}$ | true label and weak learner's prediction; product $=+1$ if right, $-1$ if wrong |
| $\alpha_t$ | vote strength of learner $t$ (large when $\varepsilon_t$ is small; 0 at $\varepsilon=0.5$) |
| $Z_t$ | normaliser so the weights sum to 1 |
| $\nu$ | learning rate / shrinkage of boosting |
| $r_i$ | residual = negative gradient of the squared loss |

**Use case:** $\varepsilon=0.2$ ⇒ $\alpha=0.693$; wrong points' weights ×2, right points ×0.5, then normalised to 0.25 / 0.0625.

---

## Q. Polynomial Regression, Regularisation, Bias-Variance

### Q1. Polynomial regression and normal equation
| Name | Function |
|---|---|
| Polynomial model | $\displaystyle\hat y=\beta_0+\beta_1x+\beta_2x^2+\dots+\beta_dx^d$ |
| Normal equation (OLS) | $\displaystyle\hat\beta=(X^TX)^{-1}X^Ty$ |

| Variable | What it is / what it does |
|---|---|
| $d$ | polynomial degree = complexity; low ⇒ underfit, high ⇒ overfit |
| $X$ | design matrix with columns $1,x,x^2,\dots,x^d$ |
| $y$ | target vector |

**Use case:** curved relationships; data $(0,1),(1,3),(2,7),(3,13),(4,21)$ ⇒ $\hat y=x^2+x+1$ ($SSE=0$) vs line $5x-1$ ($SSE=14$).

### Q2. Ridge, Lasso, Elastic Net
| Name | Function |
|---|---|
| Ridge (L2) objective | $\displaystyle\min_\beta\ \sum_i(y_i-x_i^T\beta)^2+\lambda\sum_j\beta_j^2$ |
| Ridge closed form | $\displaystyle\hat\beta=(X^TX+\lambda I)^{-1}X^Ty\quad\Big(\text{1-D: }\frac{\sum xy}{\sum x^2+\lambda}\Big)$ |
| Lasso (L1) objective | $\displaystyle\min_\beta\ \sum_i(y_i-x_i^T\beta)^2+\lambda\sum_j\lvert\beta_j\rvert$ |
| Lasso 1-D soft threshold | $\displaystyle\hat\beta=\frac{\operatorname{sign}(\sum xy)\max\big(\lvert\sum xy\rvert-\lambda/2,\ 0\big)}{\sum x^2}$ |
| Elastic Net | $\displaystyle\min\ \sum(y-x^T\beta)^2+\lambda\Big[\rho\sum\lvert\beta_j\rvert+\frac{1-\rho}{2}\sum\beta_j^2\Big]$ |

| Variable | What it is / what it does |
|---|---|
| $\lambda$ (`alpha`) | regularisation strength: 0 ⇒ OLS; large ⇒ coefficients shrink (more bias, less variance); tune by CV |
| $\beta_j$ | coefficient of feature $j$ (intercept not penalised) |
| $\rho$ | Elastic-Net mix: 1 = lasso, 0 = ridge |
| $I$ | identity matrix; adding $\lambda I$ also makes $X^TX$ invertible when features are collinear |

**Use case:** ridge for many correlated features; lasso for automatic feature selection (some $\beta=0$). Example $\sum xy=57,\sum x^2=30$: OLS 1.90; ridge($\lambda=10$) 1.425; lasso($\lambda=10$) 1.733, lasso($\lambda=114$) 0.

### Q3. Bias-variance decomposition
| Name | Function |
|---|---|
| Expected squared error | $\displaystyle E\big[(y-\hat f(x))^2\big]=\big(E[\hat f]-f\big)^2+E\big[(\hat f-E[\hat f])^2\big]+\sigma_\varepsilon^2$ |

| Variable | What it is / what it does |
|---|---|
| $\big(E[\hat f]-f\big)^2$ | **Bias²**: systematic error of the average prediction (too-simple model) |
| $E[(\hat f-E\hat f)^2]$ | **Variance**: how much predictions change between training sets (too-complex model) |
| $\sigma_\varepsilon^2$ | irreducible noise in the data |

**Use case:** diagnose under/overfitting. Model A (bias² 0, var 2) total 2; Model B (bias² 4, var 0.02) total 4.02.

---

## R. Policy-Based RL, Policy Optimisation, Planning

### R1. Policy gradient and actor-critic
| Name | Function |
|---|---|
| Objective | $\displaystyle J(\theta)=E_{\pi_\theta}\Big[\sum_t\gamma^tr_t\Big]$ |
| Policy-gradient theorem | $\displaystyle\nabla_\theta J=E_{\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,Q^\pi(s,a)\big]$ |
| Return | $\displaystyle G_t=\sum_{k\ge t}\gamma^{k-t}r_k$ |
| REINFORCE update | $\displaystyle\theta\leftarrow\theta+\alpha\,G_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ |
| Softmax-policy gradient | $\displaystyle\nabla_\theta\log\pi(a)=\mathbf 1_a-\pi$ |
| Baseline / advantage | $\displaystyle A(s,a)=Q(s,a)-V(s)\approx\delta=r+\gamma V(s')-V(s)$ |
| Actor-critic updates | $\displaystyle\theta\leftarrow\theta+\alpha\,\delta\,\nabla\log\pi_\theta(a\mid s),\qquad w\leftarrow w+\beta\,\delta\,\nabla V_w(s)$ |

| Variable | What it is / what it does |
|---|---|
| $\theta$ | policy parameters (actor) |
| $\pi_\theta(a\mid s)$ | probability of action $a$ in state $s$ |
| $G_t$ | discounted return after time $t$; scales how strongly the action is reinforced |
| $\alpha,\ \beta$ | learning rates for actor and critic |
| $\mathbf 1_a$ | one-hot vector of the chosen action |
| $V_w$, $\delta$ | critic (value network, weights $w$) and its TD error |
| baseline | subtracted from $G_t$ to cut variance without bias |

**Use case:** continuous or stochastic actions. $\pi=[0.5,0.5]$, chose $a=0$, $G=10$, $\alpha=0.1$ ⇒ $\theta=[0.5,-0.5]$, $\pi=[0.731,0.269]$.

### R2. PPO, DQN, Dyna-Q, MCTS
| Name | Function |
|---|---|
| Probability ratio | $\displaystyle r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ |
| PPO clipped objective | $\displaystyle L=E\Big[\min\big(r_tA_t,\ \operatorname{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\Big]$ |
| DQN target and loss | $\displaystyle y=r+\gamma\max_{a'}Q_{target}(s',a'),\qquad L=\big(y-Q(s,a;\theta)\big)^2$ |
| Dyna-Q planning update | $\displaystyle Q(s,a)\leftarrow Q(s,a)+\alpha\big[\hat r+\gamma\max_{a'}Q(\hat s',a')-Q(s,a)\big]$, with $(\hat r,\hat s')$ from the learned model |
| UCB1 (MCTS selection) | $\displaystyle\text{UCB}=\bar Q_i+c\sqrt{\frac{\ln N}{n_i}}$ |

| Variable | What it is / what it does |
|---|---|
| $\epsilon$ | clip range, typically 0.2: limits how far one update can move the policy |
| $A_t$ | advantage estimate |
| $Q_{target}$ | slowly updated copy of the Q-network (stabilises training) |
| $\hat r,\ \hat s'$ | reward / next state predicted by the **model** (simulated experience) |
| $N,\ n_i$ | visits to the parent node / to child $i$ |
| $c$ | exploration constant (≈ $\sqrt2$) |

**Use case:** PPO: $r=1.5,A=2$ ⇒ objective capped at 2.4. DQN: $r=1,\gamma=0.9,\max Q=5$ ⇒ $y=5.5$. Dyna-Q: $Q=50\to75\to87.5\to93.75$ in three planning steps (target 100).

---

## S. PCA Extensions

### S1. SVD link, variance retained, Kernel PCA, LDA
| Name | Function |
|---|---|
| SVD of centred data | $\displaystyle X_c=U\Sigma V^T,\qquad\lambda_i=\frac{\sigma_i^2}{N-1}$ |
| Cumulative explained variance | $\displaystyle\text{CEV}(k)=\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^D\lambda_i}$ |
| Kernel PCA | eigen-decompose the **centred** kernel matrix $\tilde K=K-1_NK-K1_N+1_NK1_N$, $K_{ij}=k(x_i,x_j)$ |
| LDA criterion (supervised) | $\displaystyle\max_w\ \frac{w^TS_Bw}{w^TS_Ww}$ |

| Variable | What it is / what it does |
|---|---|
| $V$ | columns = principal directions |
| $\sigma_i$ | singular values; $\sigma_i^2/(N-1)$ = variance along PC $i$ |
| $k,\ D$ | components kept / original dimensions |
| $1_N$ | $N\times N$ matrix with all entries $1/N$ |
| $S_B,\ S_W$ | between-class and within-class scatter matrices |

**Use case:** eigenvalues $[4,2,1,.5,.5]$ ⇒ CEV $=[.50,.75,.875,.9375,1]$ ⇒ 3 PCs for ≥ 80 %. Kernel PCA for curved structure; LDA when labels exist.

---

## T. Evaluation Extras, Cross-Validation, Tuning

### T1. Extra metrics
| Name | Function |
|---|---|
| Balanced accuracy | $\displaystyle\frac{TPR+TNR}{2}$ |
| Matthews corr. coef. (MCC) | $\displaystyle\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ |
| Adjusted $R^2$ | $\displaystyle1-(1-R^2)\frac{n-1}{n-p-1}$ |
| MAPE | $\displaystyle\frac1n\sum\Big\lvert\frac{y_i-\hat y_i}{y_i}\Big\rvert\times100\%$ |

| Variable | What it is / what it does |
|---|---|
| $TPR,\ TNR$ | recall and specificity |
| MCC range | −1 (total disagreement) … 0 (random) … +1 (perfect); uses all four confusion cells |
| $n,\ p$ | samples and predictors; adjusted $R^2$ only rises if a new feature really helps |

**Use case:** imbalanced classification (balanced accuracy, MCC). Cat-dog: 0.852 and 0.685. $R^2=0.85,n=50,p=3$ ⇒ adj. $R^2=0.840$.

### T2. Cross-validation and tuning cost
| Name | Function |
|---|---|
| CV estimate | $\displaystyle\widehat{\text{Score}}_{CV}=\frac1k\sum_{i=1}^ks_i,\qquad SE=\frac{s}{\sqrt k},\quad s=\sqrt{\frac{\sum(s_i-\bar s)^2}{k-1}}$ |
| Number of fits | k-fold: $k$; LOO: $n$; Leave-$p$-out: $\displaystyle\binom{n}{p}$; repeated k-fold: $k\times$ repeats |
| Grid search fits | $\displaystyle\Big(\prod_h\lvert V_h\rvert\Big)\times k\ \ (+\,1\text{ refit})$ |
| Nested CV fits | $\displaystyle k_{outer}\times k_{inner}\times\prod_h\lvert V_h\rvert$ |

| Variable | What it is / what it does |
|---|---|
| $s_i$ | score on fold $i$ |
| $k$ | number of folds (5 or 10 typical); small $k$ = pessimistic bias, $k=n$ = high variance |
| $\lvert V_h\rvert$ | number of candidate values for hyper-parameter $h$ |
| $SE$ | uncertainty of the CV mean |

**Use case:** accuracies $.82,.85,.80,.88,.85$ ⇒ $.840\pm.031$ (SE $.014$). Grid $3\times4\times2=24$ combos × 5 folds = 120 fits; nested 5×3×24 = 360.
