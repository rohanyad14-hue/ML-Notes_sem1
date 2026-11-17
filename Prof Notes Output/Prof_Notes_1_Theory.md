# MACSE504 Machine Learning: THEORY NOTES

<p class="cover"><b>Sections 1-19</b> are built only from the notes in <b>Prof Provided Notes</b> (slides, PDFs, DOCX). <b>Sections 20-27 (Part B)</b> fill the syllabus topics that folder does not cover. Worked examples are in <b>Prof_Notes_2_Numericals</b>; all symbols are explained in <b>Prof_Notes_3_Formula_Sheet</b>.<br>Boxes marked <b>⚠ Check</b> flag places where the professor's file contains an error or an unclear step.</p>

**Topics covered (in order):** 1 Introduction · 2 Linear & Multiple Regression · 3 Logistic Regression · 4 KNN · 5 Decision Trees · 6 Naive Bayes · 7 SVM & SVR · 8 Imbalanced Data · 9 Evaluation Metrics · 10 K-means · 11 Hierarchical Clustering · 12 Cluster Validity · 13 Association Rules (Apriori, FP-Growth) · 14 Semi-Supervised Learning · 15 Graph Learning · 16 Reinforcement Learning · 17 PCA · 18 LLE · 19 t-SNE · **Part B:** 20 Random Forest & Ensembles · 21 Polynomial Regression, Regularisation, Bias-Variance · 22 Supervised-learning challenges & applications · 23 Unsupervised/Semi-supervised types & challenges · 24 Policy-based RL, PPO, DQN, Dyna-Q, MCTS · 25 Dimensionality-reduction overview, PCA variants, limitations · 26 Metrics, Cross-Validation, Hyper-parameter Tuning, Group CV · 27 Contemporary issues

> ★ **FAT map: every syllabus module (MACSE504 v1.0) → where it is in these notes.** Sections **1-19** come from the **professor's files**. Sections **20-27 (Part B)** cover syllabus topics that **have no file in Prof Provided Notes** (random forest/ensembles, polynomial regression, regularisation, bias-variance, policy-based RL, PCA variants, cross-validation, hyper-parameter tuning, contemporary issues). Part B is built from the **syllabus + the reference book (Dangeti, *Statistics for Machine Learning*) + standard sources**, so compare it with what was taught in class.

| Module | Syllabus topics | Theory sections | Numericals | Formula sheet |
|---|---|---|---|---|
| **1** Supervised Learning | Logistic Reg., KNN, SVM, Decision Tree, **Random Forest**; Linear, **Polynomial**, Multiple Regression; spam/image apps; **overfitting, imbalanced data** | 2, 3, 4, 5, 7, 8, **20, 21, 22** | N1-N5, N6, N11, **N19-N22** | A, B, C, D, F, H, **P, Q** |
| **2** Unsupervised & Semi-supervised | Clustering (K-means, Hierarchical), Apriori, FP-Growth, evaluation metrics, Self-training, Label Spreading, Label Propagation, Naïve Bayes, SVM, Graph learning | 6, 7, 10-15, **23** | N4, N5, N7-N9, N12-N15 | E, F, I, J, K |
| **3** Reinforcement Learning | **Policy-based RL, policy gradient, policy optimisation**, value-based RL, Q-learning, model-based RL, model-based + model-free, planning | 16, **24** | N18, **N23-N25** | L, **R** |
| **4** Dimensionality Reduction | Overview, PCA (**variants, applications**), LLE, t-SNE, principles, limitations | 17, 18, 19, **25** | N16, N17, **N26** | M, N, O, **S** |
| **5** Evaluation & Tuning | Classification/regression metrics, binary & multi-class evaluation, scikit-learn metrics, **cross-validation & types, generalisation error, hyper-parameter tuning, group CV** | 9, 12, **26** | N10, **N27-N29** | G, A4, **T** |
| **6** Contemporary Issues | (no topics listed in syllabus) | **27** | – | – |

**Topic finder: every phrase in the syllabus → where to read it**

| Module | Syllabus phrase | Theory | Numericals | Formulas |
|---|---|---|---|---|
| 1 | Logistic Regression | 3 | – | B1, B2 |
| 1 | KNN | 4 | – | C1-C3 |
| 1 | SVM | 7 | N5 | F1-F4 |
| 1 | Decision Tree | 5 | N3 | D1-D4 |
| 1 | Random Forest | 20 | N19 | P1, P2 |
| 1 | Linear / Multiple Linear Regression | 2 (incl. MLR) | N1, N2 | A1-A3 |
| 1 | Polynomial Regression | 21 | N20 | Q1 |
| 1 | Spam detection, Image recognition | 22 | – | – |
| 1 | Overfitting, Imbalanced data | 5, 8, 21, 22 | N11, N20-N22 | H1, Q3 |
| 2 | Unsupervised learning types, applications, challenges | 23 | – | – |
| 2 | K-Means · Hierarchical | 10 · 11 | N7 · N8 | I1 · I2 |
| 2 | Apriori · FP-Growth | 13 | N9 | J1 |
| 2 | Evaluation metrics | 9, 12 | N10 | G1, G2, I3, I4 |
| 2 | Self-Training · Label Propagation · Label Spreading | 14 | N12 · N13 · N14 | K1, K2 |
| 2 | Naïve Bayes · SVM · Graph-based learning | 6 · 7 · 15 | N4 · N5 · N15 | E · F · K2 |
| 3 | **Policy-Based RL, Applications of Policy** | **24** | N23 | R1 |
| 3 | **Policy Gradient Methods · Policy Optimization Methods** | **24** | N23 · N24 | R1 · R2 |
| 3 | **Value-Based RL · Value Functions** | **16A** | N18, N18i | L1, L1b |
| 3 | **Q-Learning** | **16A** (also 16) | **N18a-N18j** | **L4, L4b** |
| 3 | Model-Based RL · with model-free (Dyna) · with planning | 16, 24 | N18d · N25 · N24(b) | L2 · R2 |
| 4 | Overview of dimensionality reduction | 25 | – | – |
| 4 | PCA, variants and applications | 17, 25 | N16, N26 | M1, S1 |
| 4 | LLE · t-SNE | 18 · 19 | N17 · – | N1 · O1 |
| 4 | Principles, applications, limitations | 17-19, 25 | – | – |
| 5 | Classification & regression metrics; binary, multi-class; scikit-learn metrics | 9, 26.1 | N10, N27 | G1, G2, A4, T1 |
| 5 | Cross-Validation and its types | 26.3 | N28 | T2 |
| 5 | Estimating generalisation error with CV | 26.2, 26.3 | N28(c) | T2 |
| 5 | Hyper-parameter tuning | 26.4 | N29(a) | T2 |
| 5 | Group Cross-Validation | 26.5 | N29(b) | T2 |

---

## 1. Introduction to Machine Learning (Alpaydin Ch.1)

**Definition:** programming computers to *optimise a performance criterion using example data or past experience*.

**Why learn?** (a) human expertise does not exist (navigating Mars); (b) humans cannot explain their expertise (speech recognition); (c) the solution changes over time (network routing); (d) the solution must adapt to particular cases (user biometrics). No need to "learn" payroll calculation.

**Role of statistics:** inference from a sample. **Role of computer science:** efficient algorithms to solve the optimisation problem, and to represent/evaluate the model.

| Type | Idea | Examples |
|---|---|---|
| **Association** | learn $P(Y\mid X)$ | basket analysis: $P(\text{chips}\mid\text{beer})=0.7$ |
| **Supervised** (classification, regression) | learn mapping $x\to y$ from labelled data | credit scoring, used-car price $y=wx+w_0$ |
| **Unsupervised** | learn "what normally happens"; no output | clustering, customer segmentation, colour quantisation |
| **Reinforcement** | learn a **policy** (sequence of actions) from delayed reward | game playing, robot in a maze |

**Supervised learning uses:** prediction of future cases, knowledge extraction (rule is understandable), compression (rule simpler than data), outlier detection (exceptions to the rule, e.g. fraud).
**Classification** = pattern recognition (face, character, speech, medical diagnosis, biometrics). **Regression** = continuous output (steering angle, robot-arm kinematics).

---

## 2. Linear Regression

* **Supervised, maps inputs $X$ to a continuous target $Y$.** Fits the **line of best fit** by **Ordinary Least Squares (OLS)**: minimise the sum of squared vertical distances (residuals).
* **Model:** $Y=\beta_0+\beta_1X+\varepsilon$.
  * $Y$ dependent/target; $X$ independent/predictor; $\beta_0$ intercept (value of $Y$ when $X=0$); $\beta_1$ slope (change in $Y$ per unit $X$); $\varepsilon$ random error/residual.
* **Slope** = covariance of $X,Y$ divided by variance of $X$. **Intercept** $\beta_0=\bar Y-\beta_1\bar X$ (the line always passes through $(\bar X,\bar Y)$).
* **Interpreting the slope:** positive slope = $Y$ rises with $X$; **negative slope (e.g. $-3.5$) = inverse relationship, $Y$ falls 3.5 units for each unit increase in $X$.**
### Multiple Linear Regression (MLR), applications and evaluation
* **Multiple linear regression:** $\hat y=b_0+b_1x_1+b_2x_2+\dots$. Each $b_j$ = change in $y$ per unit of $x_j$ **holding the other predictors constant**. $b_0$ = mean $y$ when all predictors are 0. Solved by the "regression sums" shortcut (see numericals) or the matrix normal equation.
* **Applications:** real-estate price, forecasting returns/spending, drug-dose response, revenue vs ad-spend.
* **Regression evaluation:** MAE, MSE, RMSE, $R^2$ (section 9).

---

## 3. Logistic Regression

* **Binary classification** with a **sigmoid** that turns a linear score into a probability in $(0,1)$. Instead of a straight line it fits an "S"-shaped curve.
* **Decision rule:** if $p\ge 0.5$ (threshold) → Class 1, else Class 0.
* **Types:** **Binomial** (2 classes), **Multinomial** (≥3 *unordered*: cat/dog/sheep), **Ordinal** (≥3 *ordered*: low/medium/high).
* **Terminology**
  * **Odds** = something occurring ÷ not occurring $=p/(1-p)$.
  * **Log-odds (logit)** = $\ln$(odds). Modelled as a *linear* combination: $\ln\frac{p}{1-p}=\beta_0+\beta_1x$.
  * **Coefficient** = estimated parameter (how a feature relates to the outcome). **Intercept** = log-odds when all features are 0.
  * **Maximum Likelihood Estimation (MLE)** = method used to estimate coefficients (maximise likelihood of the observed data).
* **Derivation of the sigmoid** (slide): $\frac{p}{1-p}=e^{z}$ with $z=w\cdot X+b$ → $p=\dfrac{e^{z}}{1+e^{z}}=\dfrac{1}{1+e^{-z}}$.
* **Loss:** linear regression uses **squared loss**; logistic regression uses **log loss (binary cross-entropy)**. Cost is $-\log h(x)$ when $y=1$ and $-\log(1-h(x))$ when $y=0$: the cost → ∞ when the model is confidently wrong, → 0 when it is right.
* Imbalanced data: use **class weights** (cost-sensitive / weighted LR), section 8.

---

## 4. K-Nearest Neighbours (KNN)

* **Idea:** "if it walks like a duck and quacks like a duck, it's a duck". Classify a test record by the **majority label of its $k$ nearest training records**. $k=1$ is the *nearest-neighbour rule*. Choose **$k$ odd** to avoid ties.
* **Alternate names:** instance-based, **lazy**, case-based, exemplar-based learning (no training phase; all work is done at query time).
* **Choice of $k$:** too small → sensitive to noise; too large → neighbourhood includes other classes. **Rule of thumb $k=\sqrt N$.**
* **Distance measures:** Euclidean, Manhattan (city-block), Minkowski ($l$-norm), Chebyshev, Mahalanobis, cosine similarity (angle between samples), Jaccard (fraction of exact matches), Camberra, correlation, chi-square, Kendall-rank. **Nominal attributes:** Value Difference Metric (VDM). **Mixed data:** heterogeneous distance functions (Wilson & Martinez, 1997).
* **Scale effects:** features with large ranges dominate (weight in kg vs protein in ng/dL). **Fix: standardise to z-scores** $z=(x-\mu)/\sigma$.
* **Regression / weighted KNN:** $\hat f(q)=\sum w_if(x_i)/\sum w_i$ (unweighted: $w_i=1$). Classification: weighted vote $\arg\max_v\sum w_i\delta(v,f(x_i))$.
* **Voronoi diagram:** each cell contains one sample; every point in the cell is closer to that sample than any other. The 1-NN decision boundary follows cell edges.
* **Problems:** (1) value of $k$; (2) choice of metric; (3) **expensive**: must compute distance to all $N$ points and store all data; (4) **curse of dimensionality**: data become sparse, Euclidean distance gives counter-intuitive results, required training data grows exponentially, partitioning degrades to linear search. (Slide example: two pairs of 12-bit vectors both have $d=1.4142$ although one pair is clearly more similar.)
* **Speed-ups:** kd-trees, Locality Sensitive Hashing (approximate distance), **condensing/editing** (remove redundant points).
  * **Decision-boundary-consistent subset:** same NN boundary as full set. **Minimum consistent set:** smallest subset classifying all original data correctly.
  * **Condensed Nearest Neighbour (CNN):** start with one sample; classify the rest with the subset; move every misclassified sample into the subset; repeat until no transfers. Incremental, **order dependent**, neither minimal nor boundary-consistent, $O(n^3)$ brute force.
  * **kd-tree:** index data in a tree, splitting on a *different dimension at each level* (e.g. $x=5$, then $y=3$ / $y=6$, ...); supports range search around the query, then linear search only in that rectangle.

---

## 5. Decision Trees

* **Origins:** ID3 (Quinlan, late 1970s), then C4.5; CART (Breiman, Friedman, Olshen, Stone) builds binary trees.
* **Induction = top-down, recursive divide-and-conquer, greedy, non-backtracking.** Start with all tuples in one node; pick the best attribute (via an *attribute-selection measure*); partition; recurse.
* **Inputs:** data partition $D$, `attribute_list`, `attribute_selection_method`.
* **Stop when:** (1) all tuples in $D$ are of one class → leaf; (2) no attributes left → **majority vote**; (3) a branch partition is empty.
* **Cost of growing:** $O(n\times|D|\times\log|D|)$ ($n$ = #attributes).

### Attribute-selection measures

| Algorithm | Measure | Splits | Pruning |
|---|---|---|---|
| **ID3** | **Information gain** (entropy) | multiway | – |
| **C4.5** | **Gain ratio** (gain ÷ split info) | multiway | **pessimistic pruning** (uses training set, no separate pruning set) |
| **CART** | **Gini index** | **strictly binary** (a discrete attribute with $v$ values has $2^v-2$ candidate binary partitions) | **cost-complexity pruning** (needs independent pruning set) |

* **Entropy / Info(D):** average information needed to identify the class of a tuple. High entropy = random; low entropy = ordered/pure.
* **Information gain** = reduction in entropy after splitting on $A$; choose the attribute with the **highest gain**. **Drawback:** biased toward attributes with many values (e.g. unique IDs give pure but useless splits).
* **Gain ratio** normalises gain by **SplitInfo** to penalise many-valued attributes. If SplitInfo→0, ratio is unstable, so C4.5 requires the chosen test's gain to be at least the **average gain** of all tests examined.
* **Gini index:** impurity of $D$; choose the split with the **largest impurity reduction** (smallest weighted Gini).

### Overfitting and pruning

* Branches reflect **noise/outliers** → overfitting. **Pruning** removes least-reliable branches → smaller/more readable trees, better accuracy on unseen data, faster, avoids subtree repetition/replication.
* **Prepruning:** stop growing early when a measure (significance, Gini...) falls below a threshold; node becomes a leaf with the majority class. *Problem: threshold choice (high → oversimplified, low → little simplification).*
* **Postpruning:** grow the full tree, then replace subtrees by a leaf (most frequent class). Slower but more reliable.
* Post-pruning metrics: **Cost complexity (CART):** $R_\alpha(T)=R(T)+\alpha|N_L|$. **Pessimistic (C4.5):** training error + a statistical penalty. **MDL:** minimise bits(tree) + bits(exceptions); "simplest solution is preferred".

### Scalability for big data
* **RainForest:** keeps an **AVC-set** (Attribute-Value, Class-label counts) per attribute per node; its size depends only on #distinct values and #classes, so it fits in memory.
* **BOAT:** bootstrap several in-memory samples → build several trees → examine & merge into a tree very close to the full-data tree.

### Practical (Colab, Iris)
`DecisionTreeClassifier(criterion="gini" | "entropy", max_depth=3, min_samples_leaf=5)`; 70/30 split; both criteria gave accuracy **95.56 %** on 45 test samples (see numericals).

---

## 6. Naïve Bayes

* **Three ways to build a classifier:** (a) model the rule directly (k-NN, trees, perceptron, SVM): *discriminative*; (b) model $P(\text{class}\mid\text{input})$ (e.g. cross-entropy): *discriminative probabilistic*; (c) model the data distribution within each class (naïve Bayes): **generative**. (b)+(c) are *probabilistic*.
* **Generative** = L separate models, each trained only on its own class's examples; can generate data by sampling. **Discriminative** = one model trained on all classes jointly.
* **Bayes rule:** Posterior = Likelihood × Prior / Evidence. **MAP rule:** assign the class with the largest posterior. $P(x)$ is common to all classes, so it is ignored: $P(c\mid x)\propto P(x\mid c)P(c)$.
* **Naïve assumption:** all features are **class-conditionally independent**, so $P(x_1..x_n\mid c)=\prod_jP(x_j\mid c)$. (The full joint is infeasible to learn.)
* **Discrete features:** *Learning* = estimate $P(c_i)$ and every $P(x_{jk}\mid c_i)$ by counting (gives $F\times L$ tables). *Test* = look up tables, multiply, MAP.
* **Continuous features:** model $P(x_j\mid c_i)$ by a **Gaussian** with class-wise mean and std; compute the density instead of table lookup.
* **Zero-conditional-probability problem:** an unseen feature value makes the whole product 0. **Fix: m-estimate** $\hat P=\dfrac{n_c+mp}{n+m}$ ($p$ = prior, usually $1/t$ for $t$ values; $m$ = number of virtual examples, up to ~1 % of the data, $m\ge1$).
* **Summary:** very fast training/testing; works well even when independence is violated; popular for **spam filtering/text classification**; good base learner for ensembles.
* **Probability basics (quiz in slides):** conditional $P(A\mid B)=P(A,B)/P(B)$; independence means $P(A,B)=P(A)P(B)$.

---

## 7. Support Vector Machines (SVM)

### Linear, hard-margin
* Separating hyperplane $w^Tx+b=0$; predict $f(x)=\operatorname{sign}(w^Tx+b)$. Many separators exist (a perceptron finds *any* one); SVM picks the **maximum-margin** one: as far as possible from both classes.
* Margin $m=2/\lVert w\rVert$, so **maximise margin ⇔ minimise $\tfrac12\lVert w\rVert^2$** subject to $y_i(w^Tx_i+b)\ge1$ (labels $y_i\in\{+1,-1\}$).
* Solved with **Lagrange multipliers $\alpha_i\ge0$** → the **dual problem** (maximise, from KKT theory): depends only on **inner products** $x_i^Tx_j$; a **quadratic programming (QP)** problem with a **global** maximum.
* **Recover:** $w=\sum\alpha_iy_ix_i$; $b$ from any support vector ($b=-\tfrac12(w^Tx^{(+1)}+w^Tx^{(-1)})$).
* **Support vectors:** points with $\alpha_i>0$ (on the margin). Most $\alpha_i=0$ → **sparse** solution; the boundary depends *only* on support vectors (moving interior points changes nothing).
* **Test:** $f(z)=\sum_{\text{SV}}\alpha_iy_i\,x_i^Tz+b$; class 1 if positive. $w$ never needs to be formed.

### Soft margin (non-separable data)
* Allow errors with **slack variables $\xi_i\ge0$**: $y_i(w^Tx_i+b)\ge1-\xi_i$. $\xi_i=0$ if no error; $\xi_i$ is an upper bound on errors.
* Minimise $\tfrac12\lVert w\rVert^2+C\sum\xi_i$. **$C$ = trade-off error vs margin.** Large $C$ → fewer errors, in the limit $C\to\infty$ = hard margin. Small $C$ → wider margin, more tolerance (more robust).
* Dual identical to hard-margin **except $0\le\alpha_i\le C$** (upper bound $C$ because $C=\alpha_i+\mu_i$ with $\mu_i,\alpha_i\ge0$). Note: the algorithm minimises the *sum of distances* of violators, not the number of errors.

### Non-linear SVM and the kernel trick
* Map $x\to\phi(x)$ into a (higher-dimensional) **feature space** where classes become linearly separable. Classic: **XOR** is not linearly separable in $(x,y)$ but is separable after adding the feature $xy$.
* Feature space may be huge/infinite, but the dual needs only inner products → **kernel $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$**; no need to compute $\phi$. A valid kernel must satisfy **Mercer's condition** (positive semi-definite kernel matrix).
* **Kernels:** polynomial $(x^Ty+1)^d$ (or $(u\cdot v)^d$); **RBF/Gaussian** $\exp(-\lVert x-y\rVert^2/2\sigma^2)$ (infinite-dimensional feature space); **sigmoid** $\tanh(\kappa x^Ty+\theta)$ (not Mercer for all $\kappa,\theta$, but still often works).
* **Building new kernels** from valid $k_1,k_2$: linear combination, exponential $\exp(k_1)$, product, polynomial of a kernel (non-negative coefficients), $f(x)k_1(x,y)f(y)$.
* Kernel = **similarity measure**; $x$ can be a string/tree/sequence (e.g. spectral kernel for DNA: counts of $l$-mers).
* **Choosing the kernel is the trickiest part.** Good start: low-degree polynomial or RBF with reasonable width. RBF-SVM ≈ RBF network with centres chosen automatically.
* **Multi-class:** change the QP, or combine binary classifiers (one-vs-all, pairwise). **Probabilities:** fit a logistic regression on SVM outputs using a held-out validation set (libsvm does this).
* **Steps:** prepare pattern matrix → choose kernel → choose kernel parameter and $C$ (software defaults or validation set) → train to get $\alpha_i$ → classify new data with $\alpha_i$ and SVs.
* **Strengths:** easy training, **no local optima** (unlike neural nets), scales to high dimensions, explicit complexity/error trade-off, non-vector inputs. **Weakness:** need to choose a good kernel.
* **Two key ideas: maximise the margin + kernel trick.**

### Support Vector Regression (SVR)
* Fit $f(x)=wx+b$ ignoring errors smaller than **$\varepsilon$** (an **ε-insensitive tube**). Points outside the tube get slack $\xi_i,\xi_i^*$ (above/below).
* Minimise $\tfrac12 w^2+C\sum(\xi_i+\xi_i^*)$; $C$ = penalty for points outside the tube. **Support vectors = points on or outside the tube boundary.**

> ⚠ **Check (SVR_Numericals.pdf):** the file says all three points are support vectors. Strictly, only points *on or outside* the tube are SVs; here only $(2,3)$ lies exactly on the lower boundary ($|{-0.5}|=\varepsilon$). Points (1,2) and (3,5) have zero error, i.e. strictly inside. Also the file only *tests* a candidate line ($w=1.5,b=0.5$); it is not proven optimal.

---

## 8. Imbalanced Data

* **Definition:** class distribution is not uniform: **majority** vs **minority** class. Models learn to predict the majority class and look accurate.
* **Examples:** credit-card fraud (99.9 % legitimate), virus detection (0.5 % positive), phishing (30 phishing in 1 M emails → "always predict not-phishing" gives **99.997 % accuracy** but is useless), medical diagnosis, hardware failure, rare events, intrusion.
* **Why classifiers fail:** most optimise error/accuracy → biased to majority. Decision trees minimise training error and pick the majority label at leaves; in k-NN the majority overwhelms the vote; perceptron updates only on mistakes (ok but slow).
* **Evaluation: accuracy is the wrong measure.** Use confusion matrix, **precision, recall, F1, PR-curve, ROC-AUC**.
  * **Precision** = correct positives ÷ predicted positives. **Recall** = correct positives ÷ actual positives.
  * *Maximise precision:* predict nothing positive. *Maximise recall:* predict everything positive → there is a **trade-off**; raise the confidence threshold → precision↑ recall↓.
  * **PR-curve:** sort by confidence, move the threshold, compute precision/recall at each break; **AUC** summarises the curve.
  * **F-measure** = weighted **harmonic** mean; F1 uses $\alpha=0.5$. The harmonic mean punishes a big gap between P and R (arithmetic mean would let a trivial classifier look good).

### Solutions
| Level | Technique | Pros | Cons |
|---|---|---|---|
| **Data** | **Random under-sampling** (remove majority) | easy, faster training | throws away information |
| | **Random over-sampling** (replicate minority) | uses all data, works broadly | overfitting, duplicate records, slower training |
| | **SMOTE** (synthetic minority over-sampling) | adds *variety*, best re-sampling results (Chawla) | over-generalisation, may amplify noise, static rate |
| | **Tomek links** (under-sampling); **SMOTE + Tomek** hybrid | cleans class overlap | |
| **Algorithm** | **Class weights / cost-sensitive learning** (weight $=$ majority%/minority% e.g. $99.997/0.003\approx33332$) | oversampling effect without the cost | needs weight-aware learner |
| | Adjust decision threshold / leaf probability estimate | | |
| | Optimise F1/AUC directly | | not all classifiers can |

* **SMOTE procedure:** for each minority sample, find its $k$ nearest *minority* neighbours; randomly pick $j$ of them; create synthetic points **along the line joining** the sample and each chosen neighbour ($j$ set by desired oversampling).
* **Weighted decision trees:** use *weighted* counts everywhere (error, majority label at a leaf).
* **Cost-sensitive logistic regression:** weight the log-loss, e.g. `class_weight={0:1, 1:20}` penalises a missed minority example 20×.
* **Always split with `stratify`** so train and test keep class proportions. Resample **only the training set**.
* Black-box view: pre/post-process data so any binary classifier can be reused.

---

## 9. Evaluation Metrics

### 9.1 Classification (confusion matrix)
* **TP** correct positive; **TN** correct negative; **FP** (Type I error) wrongly positive; **FN** (Type II error) wrongly negative. Good model: high TP/TN, low FP/FN.
* **Accuracy** only for balanced classes. **Precision** = quality of positive predictions. **Recall/Sensitivity/TPR** = coverage of actual positives. **Specificity** = TN rate. **FPR** = $1-$specificity. **PPV** = precision; **NPV** = $TN/(TN+FN)$. **F1** = harmonic mean of P and R.
* **Multi-class averaging:** **macro** = plain average of per-class metric (treats classes equally); **micro** = pool all TP/FP/FN/TN then compute (dominated by large classes); **weighted** = average weighted by class support.
* **ROC curve:** plots **TPR vs FPR** across thresholds. Perfect classifier = top-left corner; random classifier = diagonal; further up-left = better. **AUC** = area under it.

### 9.2 Regression
* **MAE** (average |error|; linear, robust to outliers). **MSE** (squares errors; penalises large errors, sensitive to outliers). **RMSE** = $\sqrt{\text{MSE}}$ (same units as target, still penalises large errors). **$R^2$** = proportion of variance explained: $1-SS_{res}/SS_{tot}$.
* *High $R^2$ does not prove a good model; it can reflect overfitting.*
* **When to use:** MAE if you want robustness to rare outliers; MSE/RMSE if large errors are especially bad.

### 9.3 Clustering (see section 12)
Silhouette and Davies-Bouldin when **ground truth is unknown**; ARI only when **true labels are available**.

> ⚠ **Check (Evaluation Metrics slides):** the cat/dog example prints F1 = 0.79 because recall was rounded to 0.85. Exact: recall $=6/7=0.857$, F1 $=0.80$.

---

## 10. K-Means Clustering

* **Partitioning clustering:** iteratively partition the data into $K$ non-empty clusters (K given in advance); the ideal partition minimises the **sum of squared distances** to each cluster's representative (centroid). The global optimum needs exhaustive search; **K-means (MacQueen, 1967) is a heuristic** that converges to stable centroids.
* **Algorithm:** (0) pick $K$ seed points (random). (1) **Assign** each object to the nearest seed/centroid (chosen distance metric). (2) **Update** centroids = mean of each cluster. (3) Repeat until **no membership changes**.
* **Complexity:** $O(tKn)$ ($t$ iterations, $K$ clusters, $n$ objects; $K,t\ll n$): efficient.
* **Weaknesses:** converges to a **local optimum** and is sensitive to initial seeds; must specify $K$; sensitive to **noise/outliers**; only **convex/spherical** clusters; needs a defined mean (categorical data?); struggles with differing densities; creates artificial groups even when no real pattern; loses accuracy in high dimensions.
* **Variants:** **K-Medoids** (robust to outliers), **K-Modes** (categorical), **CLARA** (large data), **Mixture models / EM** (uncertain membership).
* **Application:** colour-based image segmentation (H&E-stained tissue): RGB → L\*a\*b\* colour space → drop L\* → K-means (K=3) on $(a^*,b^*)$ → label pixels → separate images → threshold L\* to isolate dark-blue nuclei.
* Evaluate with WCSS/elbow, silhouette, etc. (section 12).

---

## 11. Hierarchical Clustering

* Builds a **tree (dendrogram)** of clusters.
  * **Agglomerative (AGNES, bottom-up):** each object starts as its own cluster; repeatedly **merge** the two closest clusters until one remains (or a stopping condition). Most common.
  * **Divisive (DIANA, top-down):** start with all objects in one cluster and **split** until each object is alone or a termination condition (desired #clusters, cluster diameter threshold) is met.
* **Distance between clusters (4 measures):** minimum $d_{min}$ (closest pair), maximum $d_{max}$ (farthest pair), mean $d_{mean}=|m_i-m_j|$ (distance of means), average $d_{avg}$ (mean of all pairwise distances).
* **Single linkage** (uses $d_{min}$, "nearest-neighbour"): similarity of clusters = closest pair. May give **long, thin "straggly" clusters (chaining effect)**. After merging $c_i,c_j$: $d(c_i\cup c_j,c_k)=\min(d(c_i,c_k),d(c_j,c_k))$.
* **Complete linkage** (uses $d_{max}$, "farthest-neighbour"): makes **tighter, spherical clusters**. After merging: $d(c_i\cup c_j,c_k)=\max(\cdot,\cdot)$.
* Min and max are extremes and **sensitive to outliers/noise**; mean/average distance is a compromise.
* **Procedure (distance-matrix method):** find the smallest off-diagonal entry → merge those two → rebuild the matrix with the linkage rule → repeat. The merge heights give the dendrogram.

---

## 12. Cluster Validity Measures

| Measure | Idea | Good value |
|---|---|---|
| **WCSS** (within-cluster sum of squares) | $\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$; compactness. Plot vs $K$ (**elbow method**). | small (always falls as $K$ grows, so look for the elbow) |
| **Silhouette** $s(i)$ | cohesion $a(i)$ (mean distance to own cluster) vs separation $b(i)$ (mean distance to nearest other cluster) | **+1** well placed; **0** on the boundary; **−1** probably wrong cluster. Use to choose $K$. |
| **Davies-Bouldin (DB)** | average, over clusters, of the worst ratio (scatter$_i$+scatter$_j$)/centroid-distance$_{ij}$ | **lower is better, 0 optimal**; cheaper than silhouette |
| **Calinski-Harabasz** (variance-ratio criterion) | between-cluster dispersion ÷ within-cluster dispersion | **higher is better**; best for compact, spherical clusters |
| **Adjusted Rand Index (ARI)** | agreement of predicted clusters with **true labels** over all pairs, adjusted for chance | **1** perfect; **0** random; range −1 to +1 |

Rand index $=\frac{a+d}{a+b+c+d}$ where $a,d$ = pairs treated the same way in both partitions (agreement) and $b,c$ = disagreements. **Use silhouette & DB when ground truth is unknown; use ARI only when it is known.**

---

## 13. Association Rule Mining (Apriori & FP-Growth)

* Finds **frequent patterns / associations** among items in transaction databases (market-basket analysis, cross-marketing, catalogue design). Rule form $A\Rightarrow B$ [support, confidence], e.g. `buys(x,"bread") ⇒ buys(x,"butter") [0.5%, 60%]`.
* **Support** = fraction of transactions containing $A\cup B$ $=P(A\cup B)$. **Confidence** = fraction of transactions containing $A$ that also contain $B$ $=P(B\mid A)=\text{supp}(A\cup B)/\text{supp}(A)$. **Strong rules** satisfy both minimum support and minimum confidence.
* **Two-step process:** (1) find all **frequent itemsets**; (2) generate strong rules from them.
* **Rule types:** Boolean vs quantitative; single- vs multi-dimensional; single- vs multi-level.

### Apriori
* **Apriori property:** *every subset of a frequent itemset must be frequent* (so if {A,B} is frequent, {A} and {B} are). Equivalently, a candidate with an infrequent subset can be pruned.
* **Level-wise:** $L_1$ → $C_2$ → $L_2$ → $C_3$ → ... Each pass: **join** $L_{k-1}$ with itself to get $C_k$; **prune** any candidate having an infrequent $(k-1)$-subset; **scan** DB to count; keep those with min support → $L_k$. Stop when $L_k$ is empty.
* **Candidate generation example:** $L_3=\{abc,abd,acd,ace,bcd\}$ → join gives $abcd$ and $acde$; **prune** $acde$ because $ade\notin L_3$ → $C_4=\{abcd\}$.
* **Efficiency tricks:** hash-based counting, transaction reduction, partitioning, sampling, dynamic itemset counting. **Iceberg queries** can use the same idea (compute high dimensions only if all lower ones pass the threshold).
* **Cost:** many DB scans and huge candidate sets.

### FP-Growth (no candidate generation)
1. **Scan DB once**: count single items, drop infrequent ones, **order by descending frequency** (header table).
2. **Scan again**: insert each transaction's ordered frequent items into an **FP-tree** (prefix tree; shared prefixes accumulate counts; node-links connect same items).
3. **Mine** by divide and conquer: for each item (starting from least frequent) form its **conditional pattern base** (prefix paths with that item's count) → build the **conditional FP-tree** → recurse. If the conditional tree is a **single path**, enumerate all combinations of its nodes.
* **Why FP-tree is good:** *complete* (never breaks a long pattern) and *compact* (infrequent items gone, frequent items shared; never larger than the DB).
* **Pattern-growth property:** $\alpha\cup\beta$ is frequent in DB iff $\beta$ is frequent in $\alpha$'s conditional pattern base.
* **Faster than Apriori** (an order of magnitude): no candidate generation/test, compact structure, no repeated DB scans.

---

## 14. Semi-Supervised Learning

* Train with **a little labelled + a lot of unlabelled data**. Useful when expert labelling is costly but raw data is cheap.
* **Approaches:** **Self-training** (pseudo-labels), **Co-training** (two models on different views label data for each other), **Graph-based** (labels spread over a similarity graph).
* **Pros:** saves labelling cost; better generalisation using the structure of unlabelled data. **Cons:** noisy/wrong unlabelled data hurts; careful tuning/assumptions.
* **Applications:** speech recognition (Meta AI), fraud detection (PayPal), medical imaging (few expert-marked scans).

### Self-training (loop)
1. Train a classifier (e.g. Naïve Bayes / Logistic Regression) on the labelled set $L$.
2. Predict the unlabelled set $U$ with a **confidence** score.
3. Keep only predictions above a **strict threshold** (e.g. ≥ 85 %) as **pseudo-labels**.
4. Move them from $U$ to $L$. 5. **Retrain; repeat.**
* **Risk: error propagation / confirmation bias.** A low threshold (e.g. 50 %) lets wrong pseudo-labels be learned and reinforced.

### Label Propagation (LPA)
* Graph: nodes = samples, edge weights = similarity (RBF kernel $W_{ij}=\exp(-\lVert x_i-x_j\rVert^2/2\sigma^2)$, $W_{ii}=0$).
* **Steps:** (1) initialise soft-label matrix $Y$ ($N\times C$): labelled rows one-hot, unlabelled rows 0 (or $1/C$). (2) build $W$. (3) **transition matrix $P=D^{-1}W$** (row-normalise). (4) iterate $Y\leftarrow PY$ then **clamp** labelled rows back to their original values (**hard clamping: true labels never change**). (5) final label = $\arg\max$ of each row. Like a fluid flowing from labelled "sources".

### Label Spreading (LSA)
* Graph-based, uses the **symmetric normalised Laplacian** $S=D^{-1/2}WD^{-1/2}$ and **soft clamping**: $F^{(t+1)}=\alpha SF^{(t)}+(1-\alpha)Y$, $0<\alpha<1$ (typical $\alpha=0.2$ in the notes; worked example uses 0.5).
* Each node keeps a fraction $(1-\alpha)$ of its **initial** label while absorbing its neighbours' → can override a noisy label. Iterate until $F$ changes by less than a tolerance (e.g. $10^{-3}$), then $\arg\max$.
* Four intuitive steps: **build the web** (edge weights, e.g. 0.9 vs 0.1) → **shake the web** (labels flow along edges) → **re-anchor** the truth each round → **equilibrium**.

| | Label Propagation | Label Spreading |
|---|---|---|
| Clamping | **hard** (labels locked at 100 %) | **soft** (labels may shift) |
| Graph normalisation | random-walk $D^{-1}W$ | symmetric $D^{-1/2}WD^{-1/2}$ |
| Noise robustness | poor (noisy label corrupts cluster) | high |
| Parameter $\alpha$ | none ($\alpha=1$ implicitly) | $\alpha\in(0,1)$ |

| | Self-training | Label spreading |
|---|---|---|
| Core idea | iterative model guessing | geometric graph network |
| Error-loop risk | high (confirmation bias) | low (self-correcting structure) |
| Ideal data | text, complex unstructured | images, spatial data, clusters |
| Data needs | independent rows | interconnected relationships |

---

## 15. Graph Learning

* **Traditional representation approaches**
  * **Random walk:** iterative route over the graph choosing a random edge each step. *Spotify*: nodes = songs, edge if played consecutively; a 10-step walk from a song finds related songs.
  * **Matrix factorisation:** break a big sparse matrix into low-dimensional dense matrices (embeddings). *Netflix*: user × movie rating matrix → user-profile matrix × movie-feature matrix; multiplying back fills blanks (predicts ratings).
* **Graph topologies:** **Homogeneous** (one node type, one edge type; e.g. citation network of papers, PageRank). **Heterogeneous** (many node/edge types; e.g. LinkedIn: professionals, companies, skills, jobs with "works at / possesses / requires").
* **Network tasks:** **Node classification** (predict node label from features + neighbourhood; PayPal fraud accounts). **Node clustering** (unsupervised community detection by link density, e.g. Louvain; epidemiology super-spreader groups).
* **Major GNN architectures**
  * **GCN:** convolution generalised to graphs; node update = weighted average of own and neighbours' features.
  * **GAT:** adds **attention**, learning different weights for different neighbours.
  * **GraphSAGE:** *inductive*, samples a fixed-size neighbourhood, scalable to massive/dynamic graphs.
* **Strengths:** relational awareness (non-Euclidean data), inductive generalisation to unseen nodes/graphs, combines local features with global structure. **Limitations:** **over-smoothing** (many layers make node vectors alike), scalability (neighbourhood expansion is memory heavy), vulnerability to small adversarial edge changes.
* **Applications:** recommender systems (Pinterest, Alibaba), drug discovery (atoms = nodes, bonds = edges), fraud detection, traffic/supply-chain forecasting.
* **Comparison table (slides):** Node embeddings (random walks, factorisation; homogeneous; node classification/clustering) · GNNs (message passing; attributed graphs; semi-supervised, link prediction) · Knowledge-Graph Embeddings (distance/rotation scoring; multi-relational triplets; KB completion) · Graph Transformers (global self-attention; arbitrary graphs; graph-level classification, molecular property prediction).
* **Message-passing step (GCN-style):** aggregate (mean) neighbours incl. self-loop → multiply by weight matrix $W$ → apply non-linearity (ReLU).

---

## 16. Reinforcement Learning (RL)

* An **agent** in an **environment** has a **state**, takes an **action**, receives a **reward** and the state changes. Rewards are **delayed / sparse** → **credit-assignment problem** (which earlier action earned the reward?). Learning with a **critic** (says how well you did) rather than a teacher (says what to do). Examples: chess (feedback only at the end), robot in a maze.
* **K-armed bandit (single state):** $K$ levers; learn $Q(a)$ = value of action $a$; exploit $a^*=\arg\max_aQ(a)$. Stochastic rewards: running average with the **delta rule** $Q(a)\leftarrow Q(a)+\eta[r-Q(a)]$.
* **Markov Decision Process (MDP):** state $s_t$, action $a_t$, reward $r_{t+1}\sim p(r\mid s_t,a_t)$, next state $s_{t+1}\sim P(s'\mid s_t,a_t)$; depends only on the current state and action (Markov). Has initial state(s) and possibly terminal/goal state(s). One run from start to goal = an **episode (trial)**.
* **Policy $\pi$:** map state → action. **Value $V^\pi(s)$:** expected cumulative reward following $\pi$. **Finite horizon:** sum of next $T$ rewards. **Infinite horizon:** discounted sum with **$\gamma\in[0,1)$**: $\gamma=0$ short-sighted (only immediate reward); $\gamma\to1$ far-sighted. **Optimal policy** $\pi^*$ has $V^*(s)=\max_\pi V^\pi(s)$.
* **$Q(s,a)$:** value of taking action $a$ in $s$ then behaving optimally; $V^*(s)=\max_aQ^*(s,a)$. **Bellman equation** expresses value as immediate reward + discounted value of next state. Optimal policy: $\pi^*(s)=\arg\max_aQ^*(s,a)$ (greedy at every local step gives the globally optimal sequence).

### Model-based learning (environment known)
* $P(s'\mid s,a)$ and $p(r\mid s,a)$ known → **no exploration needed**; solve by **dynamic programming**.
* **Value iteration:** initialise $V$ arbitrarily; repeat for all $s,a$: $Q(s,a)=E[r]+\gamma\sum_{s'}P(s'\mid s,a)V(s')$, $V(s)=\max_aQ(s,a)$; stop when $\max_s|V^{(l+1)}(s)-V^{(l)}(s)|<\delta$. Each iteration $O(|S|^2|A|)$. The policy may converge before the values do.
* **Policy iteration:** store/improve a **policy**: evaluate $V^\pi$ (solve linear equations) → improve $\pi'(s)=\arg\max_a(\cdot)$ → repeat until $\pi=\pi'$. Each iteration $O(|A||S|^2+|S|^3)$ (costlier) but **needs fewer iterations**.

### Model-free learning (environment unknown): Temporal Difference (TD)
* Need **exploration** to sample the unknown model. Update the value of the current state/action using the **next reward + discounted next value**; the **temporal difference** = (reward + $\gamma$ · next value) − current estimate.
* **Exploration strategies:** **ε-greedy** (prob ε random action, 1−ε best action; decrease ε over time). **Softmax** $P(a\mid s)=e^{Q(s,a)/T}/\sum_be^{Q(s,b)/T}$; **temperature $T$** high → near-uniform (explore), low → favour best (exploit); lowering $T$ gradually = **annealing**.
* **Deterministic rewards/actions:** $\hat Q(s_t,a_t)\leftarrow r_{t+1}+\gamma\max_{a'}\hat Q(s_{t+1},a')$ (a "backup"). Starting from 0, **Q values only increase, never decrease** (Alpaydin Fig 18.4: $0.9\cdot\max(0,81)=72.9\to0.9\cdot\max(100,81)=90$).
* **Non-deterministic:** keep a running average → **Q-learning (Watkins & Dayan, 1992)**: $\hat Q(s,a)\leftarrow\hat Q(s,a)+\eta[r+\gamma\max_{a'}\hat Q(s',a')-\hat Q(s,a)]$. **Off-policy** (uses the best next action regardless of the behaviour policy).
* **SARSA:** on-policy version, uses the action $a'$ actually chosen by the policy: $Q(s,a)\leftarrow Q+\eta[r+\gamma Q(s',a')-Q]$. Converges if the policy is **GLIE** (greedy in the limit with infinite exploration).
* **TD(0) for state values:** $V(s_t)\leftarrow V(s_t)+\eta[r_{t+1}+\gamma V(s_{t+1})-V(s_t)]$.
* **Eligibility traces:** keep a record of recently visited state-actions; $e_t(s,a)=1$ if just visited, else $\gamma\lambda e_{t-1}(s,a)$. **Sarsa(λ):** $Q(s,a)\leftarrow Q+\eta\,\delta_t\,e_t(s,a)$ for *all* $(s,a)$, with $\delta_t=r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)$. $\lambda=0$ → one-step (Q(0), Sarsa(0), TD(0)); $\lambda=1$ → credit to all earlier steps decays only by $\gamma$.
* **Tabular vs generalisation:** tables fail for huge/continuous spaces → use a function approximator (regression) for $Q(s,a\mid\theta)$.

### Types of model-free RL
* **Value-based** (Q-learning, DQN): learn $V$ or $Q$, act greedily. **Policy-based** (REINFORCE, policy gradient): optimise $\pi(a\mid s)$ directly; can learn *stochastic* policies. **Actor-Critic** (A2C, PPO, SAC): **actor** = policy, **critic** = value/advantage estimator that gives feedback to the actor.
* **Architectures:** *tabular* (Q-table, e.g. gridworld); *deep value networks* (DQN: network maps raw state to Q for each action); *continuous actor-critic networks* (PPO, SAC; e.g. robot-arm torque).
* **Example:** Atari Breakout with DQN: agent knows nothing about physics; learns directly from pixels (state), paddle moves (action) and score (reward).

| | Model-free | Model-based |
|---|---|---|
| Internal model | none | explicit model of transitions/rewards |
| Sample efficiency | low (millions of trials) | high (plans with simulated steps) |
| Computation | low per decision | high (simulation/search, e.g. MCTS) |
| Adaptability | slow if rewards change | fast re-planning |
| Weakness | data hunger | model exploitation (model errors compound) |

> ⚠ **Check (RL_Optimal Policy_Numerical.pdf):** it gets $Q(S,\text{Stay})=4$. That is the value of the policy "always stay" ($V=2+0.5V\Rightarrow4$). Under the optimal $V^*(S)=10$ the correct $Q^*(S,\text{Stay})=2+0.5\times10=7$. The conclusion (Leave, 10, beats Stay) is unchanged.

---

## 16A. Value-Based RL, Value Functions and Q-Learning (Module 3)

### Value-Based RL: the idea
* A **value-based** method does **not** store a policy. It learns **how good** states or state-action pairs are (a *value function*) and the policy is *implicit*: $\pi(s)=\arg\max_aQ(s,a)$ (greedy), or **ε-greedy** while learning.
* **Family:** dynamic programming (value/policy iteration, needs the model), **Monte-Carlo** (average full-episode returns), **Temporal-Difference (TD)** methods (TD(0), **Q-learning**, **SARSA**), and **DQN** (neural-network Q).
* **Pros:** simple, sample-efficient, natural for **discrete** actions. **Cons:** the $\max_a$ is hard for **continuous** actions, policy is deterministic, can be unstable with function approximation.

### Value Functions
| Function | Meaning | Definition |
|---|---|---|
| **State-value $V^\pi(s)$** | expected discounted return starting in $s$ and following $\pi$ | $E_\pi\big[\sum_{k\ge0}\gamma^kr_{t+k+1}\mid s_t=s\big]$ |
| **Action-value $Q^\pi(s,a)$** | expected return taking $a$ in $s$, **then** following $\pi$ | $E_\pi\big[\sum_{k\ge0}\gamma^kr_{t+k+1}\mid s_t=s,a_t=a\big]$ |
| **Optimal** $V^*,\ Q^*$ | best achievable under any policy | $V^*(s)=\max_\pi V^\pi(s)$, $Q^*(s,a)=\max_\pi Q^\pi(s,a)$ |
| **Advantage $A(s,a)$** | how much better $a$ is than the average action | $Q(s,a)-V(s)$ |

* **Links:** $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$; $V^*(s)=\max_aQ^*(s,a)$; $Q^\pi(s,a)=E[r]+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')$.
* **Bellman expectation equation** (value of a *given* policy): $V^\pi(s)=\sum_a\pi(a\mid s)\big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\big]$.
* **Bellman optimality equation** (value of the *best* policy): $Q^*(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')$, and $\pi^*(s)=\arg\max_aQ^*(s,a)$.
* **Why learn $Q$ instead of $V$ without a model?** With $Q$ you can choose the best action directly. With only $V$ you need $P(s'\mid s,a)$ to look one step ahead.
* **How values are estimated:** **DP** (model known, full sweeps); **Monte-Carlo** (wait for the episode to end, average returns; unbiased but high variance); **TD** (update after *every step* from a **bootstrapped** target $r+\gamma V(s')$; lower variance, online, slightly biased).

### Q-Learning
* **What it is:** a **model-free, off-policy, value-based, TD-control** algorithm (Watkins & Dayan, 1992) that learns the optimal action-value table $Q^*(s,a)$ from trial-and-error. No transition probabilities needed.
* **Q-table:** one cell per (state, action), usually initialised to 0.
* **Update rule** (after taking $a$ in $s$, seeing reward $R$ and next state $s'$):
$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[\underbrace{R+\gamma\max_{a'}Q(s',a')}_{\text{TD target}}-Q(s,a)\Big]=(1-\alpha)Q(s,a)+\alpha\,[\text{TD target}]$$
  The bracket is the **TD error**. **Deterministic case** with $\alpha=1$: $Q(s,a)=R+\gamma\max_{a'}Q(s',a')$.
* **Parameters:** $\alpha\in[0,1]$ **learning rate** (0 = never learn, 1 = overwrite); $\gamma\in[0,1)$ **discount** (how much the future matters); $\varepsilon$ **exploration rate** (decay it over time).
* **Algorithm:**
  1. Initialise $Q(s,a)=0$ for all pairs (terminal-state values stay 0).
  2. **For each episode:** start in an initial state $s$.
  3. **Repeat for each step:** choose $a$ from $s$ by **ε-greedy** on $Q$ (random with prob. $\varepsilon$, else $\arg\max_aQ(s,a)$) → take $a$, observe $R$, $s'$ → apply the update rule → $s\leftarrow s'$.
  4. Stop the episode at a terminal state (for it, $\max_{a'}Q=0$). Repeat for many episodes.
  5. **Extract the policy:** $\pi^*(s)=\arg\max_aQ(s,a)$.
* **Off-policy meaning:** the **behaviour policy** (ε-greedy, explores) generates the data, but the update uses the **greedy target policy** ($\max_{a'}$). So it learns the optimal policy *while* behaving exploratively.
* **Convergence (tabular):** $Q\to Q^*$ with probability 1 if every $(s,a)$ is visited infinitely often and $\alpha$ is decayed ($\sum\alpha_t=\infty,\ \sum\alpha_t^2<\infty$). In the **deterministic** case starting from 0 the $Q$ values **only increase** toward $Q^*$ and never decrease (Alpaydin Fig 18.4).
* **Reward propagates backwards:** a state $N$ steps from the goal only gets its value after about $N$ episodes: $Q=R\gamma^{N-1}$, e.g. 100, 90, 81, 72.9 for $\gamma=0.9$ (Numerical N18c).

**How to solve a Q-learning exam problem (checklist):**
1. Write the update rule with the given $\alpha,\gamma$. 2. For the **current** pair take the old $Q(s,a)$. 3. Find the next state $s'$ and compute $\max_{a'}Q(s',a')$ (**0 if terminal**). 4. TD target $=R+\gamma\max Q'$. 5. New $Q=(1-\alpha)Q_{old}+\alpha\,\text{target}$. 6. Update **only that one cell**; repeat in the order of the trajectory. 7. Final policy = row-wise $\arg\max$.

| | **Q-learning** | **SARSA** |
|---|---|---|
| Policy type | **off-policy** | **on-policy** |
| Next-step value used | $\max_{a'}Q(s',a')$ (greedy best) | $Q(s',a')$ of the action **actually chosen** by the behaviour policy |
| Behaviour | learns the optimal path even if it explores | learns the value of the exploring policy (safer near danger, e.g. cliff) |
| Update | $R+\gamma\max_{a'}Q(s',a')$ | $R+\gamma Q(s',a')$ |

* **Limitations:** the table explodes for large/continuous state spaces → **DQN** (neural net + experience replay + target network); **maximisation bias** (overestimates values) → **Double Q-learning**; continuous actions → policy-based / actor-critic methods (section 24).
* **Applications:** grid-world/robot navigation, game playing (Atari via DQN), traffic-signal control, resource allocation, recommendation, inventory/pricing decisions.

---

## 17. Principal Component Analysis (PCA)

* **Linear, unsupervised** dimensionality reduction: finds orthogonal directions (principal components) of **maximum variance**; project data onto the top few. Best for linear structure, preserves global variance.
* **Pipeline:** (1) **mean-centre** the data; (2) compute the **covariance matrix** $S$ ($D\times D$); (3) solve the **characteristic equation** $\det(S-\lambda I)=0$ for eigenvalues $\lambda$ (variance along each new axis); (4) find **eigenvectors** from $(S-\lambda I)v=0$, normalise to unit length (**PC1 = eigenvector of largest $\lambda$**); (5) **project**: score $=X_{centred}\cdot u$; (6) optional **reconstruction** $=\text{score}\cdot u^T+\text{mean}$ (information loss = dropped components).
* Covariance uses divisor $N$ (population) in the 5-point slide and $N-1$ (sample) in the handwritten solved example. Say which one you use.
* **Explained variance ratio** $=\lambda_i/\sum\lambda$. **Scree plot** (explained variance per PC). **Choosing #PCs:** keep enough to reach ~**80 %** cumulative variance (the handwritten note's example: 0.60+0.25+0.15=0.90 → 3 PCs are enough). **PCA scores are what you feed to classification/regression.**
* Eigenvectors are orthogonal ($n_1\perp n_2$); PC1 is the line of maximum spread.

---

## 18. Locally Linear Embedding (LLE)

* **Non-linear, unsupervised manifold learning.** Assumption: data lie on/near a smooth low-dimensional **manifold**, and each point with its neighbours lies on a **locally linear patch**. Goal: map $X\in\mathbb R^D$ to $Y\in\mathbb R^d$ while **preserving local neighbourhood relationships** (e.g. unroll a Swiss roll). Roweis & Saul (2000).
* **Step 1: neighbours.** Find $K$ nearest neighbours (Euclidean). *Small K loses topology; large K loses local linearity.*
* **Step 2: reconstruction weights $W$.** Minimise $\sum_i\lVert X_i-\sum_jW_{ij}X_j\rVert^2$ subject to $W_{ij}=0$ for non-neighbours and **each row sums to 1**. Solution via local Gram/covariance matrix $C$: $W_i=C^{-1}\mathbf 1/(\mathbf 1^TC^{-1}\mathbf 1)$ (Lagrange multipliers).
* **Step 3: embedding $Y$.** Keep $W$ fixed, minimise $\sum_i\lVert Y_i-\sum_jW_{ij}Y_j\rVert^2=\operatorname{Tr}(YMY^T)$ with $M=(I-W)^T(I-W)$ (sparse, symmetric), constraints $\sum Y_i=0$ and $\tfrac1NYY^T=I$. Take the **bottom eigenvectors of $M$**: **discard the smallest (eigenvalue 0, constant vector $[1,\dots,1]^T$)**, use the next $d$.
* Properties of $M$: symmetric; rows/columns sum to 0.

---

## 19. t-SNE (t-Distributed Stochastic Neighbour Embedding)

* **Non-linear, unsupervised, probability-based**, van der Maaten & Hinton (2008). Focuses on preserving **small pairwise distances (local structure)**, not variance. Best for **visualising** high-dimensional data in 2D/3D (shows clusters/sub-clusters).
* **Step 1 (high-D):** similarity of $x_j$ to $x_i$ as a **conditional probability $p_{j\mid i}$** from a **Gaussian** centred at $x_i$ (near → high, far → ≈0). **Perplexity** (typically 5–50) = effective number of neighbours; sets the local scale $\sigma_i$.
* **Step 2 (low-D):** similarities $q_{ij}$ with a **Student-t distribution (1 degree of freedom = Cauchy)**. Its **heavy tails** let moderately distant points sit further apart → solves the **crowding problem**.
* **Step 3:** minimise the **KL divergence** $C=KL(P\Vert Q)=\sum p_{ij}\log(p_{ij}/q_{ij})$ by **gradient descent**, moving the low-D points until stable.
* **Limitations:** computationally intensive; mainly for visualisation, **not feature extraction**.

### PCA vs LLE vs t-SNE
| Method | Type | Use when |
|---|---|---|
| **PCA** | linear, global | data linear; keep global variance & large distances |
| **LLE** | non-linear, local | preserve local neighbourhood geometry; unroll manifolds |
| **t-SNE** | non-linear, probabilistic | visualise complex data in 2D/3D; reveals clusters |

<div class="pb"></div>

# PART B: Syllabus Topics Not Covered by the Professor's Files

> ⚠ **Source note:** sections 20-27 are **not from Prof Provided Notes** (that folder has nothing on them). They follow the MACSE504 syllabus and standard references (Dangeti *Statistics for Machine Learning*; Müller *Intro to ML with Python*; Sutton & Barto; scikit-learn docs). Every number in the matching numericals was recomputed in Python/scikit-learn. Check wording against your class notes.

---

## 20. Ensemble Methods and Random Forest (Module 1)

* **Ensemble idea:** combine many models so errors cancel. A single deep tree has **low bias, high variance**; averaging many decorrelated trees cuts the variance.
* **Bootstrap sample:** draw $n$ rows **with replacement** from $n$. Probability a given row is *never* picked $=(1-\tfrac1n)^n\to e^{-1}\approx0.368$, so each bootstrap contains about **63.2 %** of the distinct rows. The left-out ~36.8 % are the **out-of-bag (OOB)** rows.

| Method | How it works | Reduces | Key facts |
|---|---|---|---|
| **Bagging** (bootstrap aggregating) | train each model on a bootstrap sample, **parallel**; classification = majority vote, regression = average | **variance** | works best with unstable, low-bias models (deep trees) |
| **Random Forest** | bagging of trees **plus random feature subset at every split** ($m\approx\sqrt p$ for classification, $p/3$ for regression) | variance (decorrelates trees) | **OOB error** = free validation estimate; **feature importance** (mean decrease in Gini/impurity or permutation); handles imbalance via `class_weight`; little tuning needed |
| **AdaBoost** | models trained **sequentially**; misclassified points get **higher weight** next round; final = weighted vote | **bias** | each learner's weight $\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$; sensitive to noisy labels |
| **Gradient boosting** | each new tree fits the **residuals (negative gradient of the loss)** of the current ensemble, scaled by learning rate $\nu$ | bias | **XGBoost** adds regularisation, shrinkage, parallel split search |
| **Stacking** | base models' predictions become inputs to a **meta-learner** | both | needs out-of-fold predictions to avoid leakage |

* **Why decorrelation matters:** variance of the average of $B$ trees with variance $\sigma^2$ and pairwise correlation $\rho$ is $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$. More trees ($B\uparrow$) kills the second term only; random feature selection lowers $\rho$.
* **Random Forest vs single tree:** more accurate, far less overfitting, but **less interpretable** and slower.
* **Main hyper-parameters:** `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf`, `class_weight`.
* **Applications (syllabus):** spam detection (text features → classifier), image recognition (pixels/features → classes), fraud detection with class imbalance (experiment 2).

---

## 21. Polynomial Regression and Regularisation (Module 1)

### Polynomial regression
* Model $\hat y=\beta_0+\beta_1x+\beta_2x^2+\dots+\beta_dx^d$. It is **still linear regression**, because it is linear in the *coefficients*; the powers of $x$ are just new features. Solve by least squares / normal equation $\beta=(X^TX)^{-1}X^Ty$.
* **Degree $d$ is the complexity knob:** too low → **underfitting** (high bias); too high → **overfitting** (wiggly curve through noise; high variance). Choose $d$ by validation / cross-validation. **Scale features** first (powers blow up).

### Regularisation (shrinkage) fights overfitting by penalising large coefficients
| | **Ridge (L2)** | **Lasso (L1)** | **Elastic Net** |
|---|---|---|---|
| Penalty added to RSS | $\lambda\sum\beta_j^2$ | $\lambda\sum\lvert\beta_j\rvert$ | $\lambda\big[\rho\sum\lvert\beta_j\rvert+\tfrac{1-\rho}{2}\sum\beta_j^2\big]$ |
| Effect | **shrinks** all coefficients toward 0, never exactly 0 | **sets some coefficients exactly 0** → built-in **feature selection**, sparse model | mix of both |
| Closed form | $\beta=(X^TX+\lambda I)^{-1}X^Ty$ | none (soft-thresholding / coordinate descent) | none |
| Good when | many correlated features | few important features | correlated groups + selection |
* $\lambda$ (sklearn `alpha`) is a **hyper-parameter**: $\lambda=0$ → plain OLS; $\lambda\to\infty$ → all coefficients → 0 (high bias). Pick it by cross-validation. **Standardise** features first. **The intercept is not penalised.**
* Geometry: ridge constraint region is a circle (no corners); lasso region is a diamond whose corners lie on the axes, so the optimum often lands on an axis (a coefficient = 0).

### Bias-variance trade-off
$$E\big[(y-\hat f(x))^2\big]=\underbrace{\text{Bias}^2}_{\text{systematic error}}+\underbrace{\text{Variance}}_{\text{sensitivity to training set}}+\underbrace{\sigma^2_\varepsilon}_{\text{irreducible noise}}$$

| | High bias (underfit) | High variance (overfit) |
|---|---|---|
| Symptom | train error **high**, validation error high | train error **low**, validation error **much higher** |
| Cause | model too simple, too much regularisation | model too complex, too little data/regularisation |
| Fix | more features / higher degree / lower $\lambda$ / boosting | more data, regularise, prune, bagging/random forest, fewer features |

* Increasing model complexity: bias ↓, variance ↑. The best model sits at the minimum of the validation-error U-curve.

---

## 22. Challenges and Applications of Supervised Learning (Module 1)
* **Overfitting** (see sections 5, 21), **imbalanced data** (section 8), **noisy / missing / mislabelled data**, **feature scaling**, **curse of dimensionality** (KNN, SVM), **interpretability vs accuracy** (linear/tree vs ensembles/deep nets), **data leakage** (information from test data used in training, e.g. fitting a scaler before splitting).
### Applications: Spam Detection and Image Recognition
* **Spam detection:** text → bag-of-words/TF-IDF → Naïve Bayes / Logistic Regression / SVM; imbalanced, so judge by precision/recall/F1.
* **Image recognition:** pixels → features (or CNN) → SVM / Random Forest / KNN; high dimensional → PCA first.

---

## 23. Unsupervised and Semi-Supervised Learning: Types, Applications, Challenges (Module 2)
* **Types of unsupervised learning:** (1) **clustering** (K-means, hierarchical, DBSCAN, GMM); (2) **association-rule mining** (Apriori, FP-Growth); (3) **dimensionality reduction** (PCA, LLE, t-SNE); (4) **anomaly/outlier detection**.
* **Applications:** customer segmentation, market-basket analysis, image/colour segmentation, document grouping, anomaly/fraud detection, data visualisation/compression.
* **Challenges of unsupervised learning:** no ground truth, so evaluation is indirect (silhouette, DB, WCSS); choosing $K$ and a distance measure; scaling; high dimensions; sensitivity to initialisation/outliers; results are subjective to interpret.
* **Semi-supervised assumptions (why unlabeled data helps):** **smoothness** (near points share labels), **cluster / low-density separation** (decision boundary should avoid dense regions), **manifold** (data lie on a low-dimensional manifold). If an assumption is wrong, unlabeled data **can hurt**.
* **Co-training:** two classifiers trained on **two different views (feature subsets)** of the data; each labels its most confident unlabeled examples for the other. Requires views that are sufficient and conditionally independent.
* **Challenges:** noisy pseudo-labels (error propagation), threshold/hyper-parameter tuning, graph construction cost ($O(N^2)$ similarity matrix), class imbalance in the labelled set.

---

## 24. Policy-Based RL, Policy Optimisation and Planning (Module 3)

### Policy gradient methods
* **Policy-based** methods learn a **parameterised policy $\pi_\theta(a\mid s)$** directly (a probability over actions) and do **gradient ascent** on the expected return $J(\theta)=E_{\pi_\theta}\big[\sum_t\gamma^tr_t\big]$.
* **Why policy-based?** handles **continuous / high-dimensional actions** (robot torque), learns **stochastic** policies (needed under partial observability, games like rock-paper-scissors), smoother changes in behaviour than greedy $\arg\max$. **Cons:** high-variance gradients, sample-inefficient, converge to local optima.
* **Policy-gradient theorem:** $\nabla_\theta J=E_{\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\;Q^{\pi}(s,a)\big]$. The **log-derivative trick** turns the gradient of an expectation into an expectation of a gradient that can be sampled.
* **REINFORCE (Monte-Carlo policy gradient):** run a full episode, compute the return $G_t=\sum_{k\ge t}\gamma^{k-t}r_k$, update $\theta\leftarrow\theta+\alpha\,G_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)$. Increases the probability of actions followed by high return. For a **softmax policy** $\nabla_{\theta}\log\pi(a)=\mathbf 1_a-\pi$ (one-hot of the chosen action minus the probability vector).
* **Baseline:** subtract $b(s)$ (usually $V(s)$) from $G_t$ to **reduce variance** without adding bias.

### Applications of Policy-Based RL
* **Robotics / control:** continuous joint torques, grasping, locomotion (actions are real numbers, so $\arg\max_a$ is impractical). **Games:** Go, Dota, StarCraft (policy-gradient/PPO agents). **Autonomous driving:** steering and acceleration. **Finance:** portfolio/trading policies. **Language models:** PPO in RLHF to align outputs with human preferences. **Stochastic policies** also suit partially observable or adversarial settings (mixed strategies).

### Actor-critic
* **Actor** = policy $\pi_\theta$; **critic** = value function $V_w$. The critic's **TD error** $\delta=r+\gamma V(s')-V(s)$ (an estimate of the **advantage** $A(s,a)=Q(s,a)-V(s)$) replaces $G_t$: $\theta\leftarrow\theta+\alpha\,\delta\,\nabla\log\pi_\theta(a\mid s)$ and $w\leftarrow w+\beta\,\delta\,\nabla V_w(s)$. Lower variance than REINFORCE, can update every step. Variants: A2C/A3C, **PPO**, **SAC**, DDPG.

### Policy optimisation methods (keep updates stable)
* **TRPO:** maximise the surrogate objective subject to a **KL-divergence trust region** between old and new policy (stable, complex).
* **PPO (Proximal Policy Optimisation):** simpler. With ratio $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{old}}(a_t\mid s_t)$ maximise the **clipped objective** $L=E\big[\min\big(r_tA_t,\ \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\big]$ ($\epsilon\approx0.2$). Clipping stops the new policy moving too far from the old one in a single update.

### Value-based RL recap and Deep Q-Network (DQN)
* **Value-based** methods learn $V$ / $Q$ and act greedily (Q-learning, section 16). **DQN** replaces the Q-table by a neural network and stabilises it with an **experience-replay buffer** (random past transitions, breaks correlation) and a **target network** (slowly updated copy): target $y=r+\gamma\max_{a'}Q_{target}(s',a')$, loss $(y-Q(s,a))^2$.
* **Policy-based vs value-based:** value-based = discrete actions, deterministic greedy policy, usually more sample-efficient; policy-based = continuous actions, stochastic policies, directly optimises what you care about.

### Model-based RL, combining with model-free, and planning
* **Model-based RL** learns (or is given) a model $\hat P(s'\mid s,a),\hat R(s,a)$ and **plans** with it. **Model-free** learns from real experience only.
* **Model-based + model-free (Dyna):** use real experience **both** to update the value function directly (model-free) **and** to improve the model; then run extra **planning updates** from simulated experience generated by the model. **Dyna-Q:** after every real step, do $n$ planning steps: pick a previously seen $(s,a)$, query the model for $(r,s')$, apply the Q-learning update. More planning steps → faster learning per real sample.
* **Model-based with planning:** plan **at decision time** by simulating futures: **value iteration / dynamic programming** (known model), **trajectory optimisation/MPC** (continuous control), **Monte-Carlo Tree Search (MCTS)**: repeat (1) **selection** by UCB $\bar Q+c\sqrt{\ln N/n}$, (2) **expansion**, (3) **simulation/rollout**, (4) **back-propagation** of the result. Used in AlphaGo.
* **Trade-offs:** model-based = sample-efficient, adapts quickly, but suffers **model bias** (errors compound); model-free = simple, robust, but data-hungry (see the comparison table in section 16).

---

## 25. Dimensionality Reduction: Overview, PCA Variants, Limitations (Module 4)

* **Why reduce dimensions?** curse of dimensionality (distances lose meaning, data sparse, overfitting), faster training, less storage, noise removal, **visualisation (2-D/3-D)**.
* **Two routes:** **feature selection** (keep a subset of original features: filter / wrapper / embedded e.g. Lasso, tree importance) vs **feature extraction** (build new features: PCA, LDA, LLE, t-SNE, autoencoders).
* **Linear vs non-linear:** PCA, LDA, (classical MDS) are linear; Kernel PCA, Isomap, LLE, t-SNE, UMAP are non-linear. **Supervised vs unsupervised:** **LDA** uses class labels (maximise between-class / within-class scatter); PCA ignores labels.
* **PCA and SVD:** for centred data $X_c=U\Sigma V^T$; the principal directions are the columns of $V$ and eigenvalues $\lambda_i=\sigma_i^2/(N-1)$. (Checked on the professor's hand example: singular values 9.55, 4.45 ⇒ $\lambda=30.38,\,6.62$.) scikit-learn's PCA uses SVD.
* **Choosing the number of components:** cumulative explained variance ≥ ~80-95 %, scree-plot elbow, or Kaiser rule (keep $\lambda>$ average eigenvalue; for standardised data $\lambda>1$).
* **Standardise first** (or use the correlation matrix) when features have different units, otherwise the large-variance feature dominates PC1.

| PCA variant | Idea | When |
|---|---|---|
| **Standard PCA** | eigen/SVD of covariance | small–medium data fit in memory |
| **Randomised PCA** | randomised SVD approximation | large data, few components wanted |
| **Incremental PCA** | processes data in mini-batches | data too big for memory / streaming |
| **Kernel PCA** | PCA in a kernel-induced feature space (RBF, polynomial); eigen-decompose the centred kernel matrix | **non-linear** structure |
| **Sparse PCA** | adds L1 penalty so loadings are sparse | interpretable components |
| **Probabilistic PCA / Factor analysis** | latent-variable (Gaussian) model with noise term | missing data, density estimate |
| **PCA whitening** | scale scores to unit variance | decorrelate inputs for downstream models |

* **Applications:** MNIST/image compression and denoising (keep ~95 % variance with far fewer dims), face recognition (eigenfaces), visualisation, preprocessing before KNN/SVM/clustering, genomics, finance (factor risk).
* **Limitations of PCA:** only **linear**; assumes variance = importance (a high-variance direction may be noise, a low-variance one may separate classes); sensitive to scaling and outliers; components are hard to interpret; unsupervised (ignores labels).
* **Limitations of LLE:** sensitive to $K$ and noise, needs dense, evenly sampled manifold, weights unstable when $K>d$ (needs regularisation), no natural out-of-sample mapping, doesn't preserve global structure.
* **Limitations of t-SNE:** $O(N^2)$ cost (heavy for big data), **stochastic** (different runs differ), **perplexity-sensitive**, **cluster sizes and between-cluster distances are not meaningful**, no out-of-sample transform, not for feature extraction.

---

## 26. Model Evaluation, Cross-Validation and Hyper-parameter Tuning (Module 5)

### 26.1 Extra metrics and scikit-learn
| Metric | Meaning | sklearn |
|---|---|---|
| Accuracy, precision, recall, F1 | see section 9 | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report` |
| Confusion matrix | counts TP/FP/FN/TN | `confusion_matrix` |
| ROC-AUC, PR-AUC | threshold-free ranking quality | `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score` |
| **Balanced accuracy** | $(TPR+TNR)/2$; fair under imbalance | `balanced_accuracy_score` |
| **MCC** (Matthews corr. coef.) | one number from all four cells, −1 to +1; reliable on imbalanced data | `matthews_corrcoef` |
| Log loss | penalises confident wrong probabilities | `log_loss` |
| MAE / MSE / RMSE / $R^2$ | regression, section 9.2 | `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `r2_score` |
| **Adjusted $R^2$** | $R^2$ corrected for number of predictors $p$ (does not rise just by adding features) | compute manually |
| MAPE, median AE | % error, outlier-robust error | `mean_absolute_percentage_error`, `median_absolute_error` |
| Multi-class | `average='macro' / 'micro' / 'weighted'` argument; `multi_class='ovr'` for AUC | |

* **Binary classification:** pick the metric from the cost of errors (disease screening → recall; spam filter → precision; imbalance → F1/PR-AUC/MCC). **Multi-class:** confusion matrix is $K\times K$; per-class precision/recall then macro/micro/weighted. **Regression:** MAE for robustness, RMSE if large errors are costly, $R^2$ for explained variance.
* **Scoring in CV:** `cross_val_score(model, X, y, cv=5, scoring='f1')`; error metrics use the **negative** convention (`neg_mean_squared_error`) so that "bigger is better".

### 26.2 Why cross-validation
* Training error is **optimistic**. We need an estimate of the **generalisation error** on unseen data. A single hold-out split is **high variance** (depends on which rows landed in test) and wastes data. **Cross-validation (CV)** reuses all data for both training and testing across several splits and averages.
* **Hold-out:** one train/val/test split. **Never tune on the test set**: train → tune on validation (or CV) → report once on the untouched test set.

### 26.3 Cross-validation schemes
| Scheme | How | Use / notes |
|---|---|---|
| **k-fold** | split into $k$ equal folds; each fold is the test set once; average $k$ scores | default, $k=5$ or $10$; $k$ fits |
| **Stratified k-fold** | each fold keeps the **class proportions** | classification, esp. **imbalanced** (sklearn default for classifiers) |
| **Repeated k-fold** | repeat k-fold with different shuffles | more stable estimate, $k\times$ repeats fits |
| **Leave-One-Out (LOO)** | $k=n$; one sample is the test set each time | tiny data; $n$ fits, low bias, **high variance**, expensive |
| **Leave-P-Out** | all $\binom{n}{p}$ subsets of size $p$ as test | rarely used (explodes) |
| **Shuffle-split / Monte-Carlo CV** | random train/test splits repeated | control train size independent of #splits |
| **Time-series split** (forward chaining) | train on the past, test on the next block; training window expands | temporal data (never shuffle: avoids look-ahead leakage) |
| **Group k-fold / Leave-One-Group-Out** | **all rows of a group stay in the same fold** | grouped/correlated rows (same patient, user, site, image source) |
| **Nested CV** | outer loop estimates performance, inner loop tunes hyper-parameters | unbiased estimate when tuning |

* **Choice of $k$ (bias-variance of the estimator):** small $k$ (e.g. 2-3) → models trained on less data → **pessimistic (biased) estimate**, low variance; large $k$/LOO → nearly unbiased but **higher variance** and costly. $k=5$/$10$ is the practical compromise.
* **Estimating generalisation error with CV:** $\widehat{\text{Err}}_{CV}=\frac1k\sum_{i=1}^ks_i$ (mean fold score); report the **standard deviation** across folds and standard error $s/\sqrt k$ as the uncertainty.
* **Group CV:** if rows from the same group appear in both train and test, the model memorises group-specific traits and the score is **over-optimistic (leakage)**. GroupKFold guarantees each group is wholly in train or wholly in test; Leave-One-Group-Out tests on one unseen group at a time (mimics deployment on a new patient/site).
* **Avoid leakage:** do scaling/imputation/feature selection **inside** each training fold (use a scikit-learn `Pipeline`), never on the full data first.

### 26.4 Hyper-parameter tuning
* **Parameters** are learned from data (weights, split thresholds). **Hyper-parameters** are set *before* training and control the model (k in KNN, $C,\gamma$ in SVM, tree depth, $\lambda$, learning rate, number of trees).
* **Grid search** (`GridSearchCV`): try **every combination** of a user grid with CV; picks best mean CV score, then **refits** on all training data. #fits $=\prod(\text{values per hyper-parameter})\times k$. Exhaustive but explodes ("curse of dimensionality" of the grid).
* **Random search** (`RandomizedSearchCV`): sample `n_iter` random combinations (from distributions). Usually finds a good setting with far fewer fits because **only a few hyper-parameters really matter** (Bergstra & Bengio), and it can explore continuous ranges.
* **Smarter search:** **successive halving** (`HalvingGridSearchCV`: start many candidates on a small budget, keep the best, give them more data), **Bayesian optimisation** (uses past results to pick the next trial; Optuna/Hyperopt), evolutionary search, **AutoML**.
* **Workflow:** split off a test set → define pipeline (preprocessing + model) → grid/random search with CV on the training set → inspect `best_params_`, `best_score_` → evaluate the refit model **once** on the test set. Use **nested CV** to get an unbiased score of the whole tuning procedure.
* **Diagnostics:** **validation curve** (score vs one hyper-parameter: shows under/over-fit region) and **learning curve** (score vs training-set size: if train and validation scores converge low → high bias → need a richer model; big gap → high variance → more data/regularise).

---

### 26.5 Group Cross-Validation (summary)
* **Problem:** rows are **not independent**: several rows come from the same patient / user / site / image source. A random split puts a patient's rows in both train and test, so the model "recognises the patient" and the score is **over-optimistic (leakage)**.
* **Solution:** pass a **`groups`** array; every group is kept **entirely in train or entirely in test**. `GroupKFold(n_splits)` (folds of whole groups), `LeaveOneGroupOut` (test on each group in turn), `StratifiedGroupKFold` (also keeps class ratios), `GroupShuffleSplit`. Usage: `cross_val_score(model, X, y, groups=g, cv=GroupKFold(5))`.
* **Rule:** the CV scheme must mimic deployment. New patients at test time ⇒ group CV; future data ⇒ time-series split. Worked example: Numericals N29(b).

---

## 27. Contemporary Issues (Module 6, 2 hours; the syllabus lists no topics)
*Typical items an examiner could ask; treat as general knowledge.*
* **Responsible AI:** **bias and fairness** (skewed data → discriminatory models; fairness metrics, balanced sampling), **explainability** (LIME, SHAP, feature importance, interpretable models), **privacy** (differential privacy, **federated learning**), **robustness / adversarial examples**, **accountability and regulation** (data-protection laws, model cards).
* **Technology trends:** **deep learning / transformers / large language models and foundation models**, generative AI, self-supervised learning (learn from unlabeled data, link to Module 2), **AutoML**, **MLOps** (deployment, monitoring, **data and concept drift**, retraining), **edge/TinyML**, energy cost and **Green AI**, reinforcement learning from human feedback.
