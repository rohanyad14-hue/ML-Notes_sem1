# MACSE504: Machine Learning - Module 2
# Comprehensive Study Notes: Unsupervised and Semi-Supervised Learning

<div class="meta-box">
<strong>Course:</strong> MACSE504 Machine Learning (ETH 1.0) | School of Computer Science and Engineering<br>
<strong>Prescribed Textbooks:</strong><br>
1. L. Goel, <em>Artificial Intelligence: Concepts and Applications</em>, Wiley, 2021.<br>
2. A. Müller & S. Guido, <em>Introduction to Machine Learning with Python</em>, O'Reilly, 2016.<br>
3. T. Jo, <em>Machine Learning Foundations: Supervised, Unsupervised, and Advanced Learning</em>, Springer Nature, 2021.<br>
4. P. Dangeti, <em>Statistics for Machine Learning</em>, Packt Publishing, 2017.<br>
<strong>VIT Exam Target:</strong> 10-Mark Detailed Answers for K-Means Convergence, Hierarchical Linkages, Apriori vs FP-Growth, Self-Training & Label Propagation/Spreading.
</div>

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **CampusX (Nitish Singh) - K-Means Clustering Math & Elbow Method (19:10 min):** [Watch on YouTube](https://www.youtube.com/watch?v=QXOkPvFM6NU)  
  *Core Focus:* WCSS derivation, Lloyd's optimization algorithm, and K-Means++ initialization.
* **Krish Naik - Hierarchical Clustering Agglomerative & Dendrogram (16:45 min):** [Watch on YouTube](https://www.youtube.com/watch?v=7xHsRkOdVwo)  
  *Core Focus:* Proximity matrix updates, Single vs Complete vs Average Linkage, cutting dendrograms.
* **Gate Smashers (Varun Singla) - Apriori Algorithm with Solved Example (14:30 min):** [Watch on YouTube](https://www.youtube.com/watch?v=guVvtZ7ZClI)  
  *Core Focus:* Candidate itemset generation, pruning via downward closure property, support & confidence.
* **CampusX - Semi-Supervised Learning: Label Propagation & Spreading (18:25 min):** [Watch on YouTube](https://www.youtube.com/watch?v=Nq39y2pYyZs)  
  *Core Focus:* Affinity matrices, normalized Graph Laplacians, and clamping parameters.

---

## 1. Clustering Foundations

<div class="diagram-box">
  <img src="images/ml_kmeans_elbow.svg" style="max-width:560px;" alt="K-Means Clustering and Elbow Curve">
  <div class="diagram-caption">Figure 2.1: K-Means Clustering Voronoi tessellations and Within-Cluster Sum of Squares (WCSS) Elbow heuristic.</div>
</div>

### 1.1 K-Means Clustering Formulation (Lloyd's Algorithm)
Given dataset $\mathcal{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_n\} \subset \mathbb{R}^d$, partition into $K$ disjoint clusters $\mathcal{C} = \{C_1, \dots, C_K\}$ minimizing the Within-Cluster Sum of Squares (WCSS / Inertia):
$$J(\mathcal{C}, \boldsymbol{\mu}) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$
where centroid $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$.

#### Iterative Optimization Steps:
1. **Assignment Step (Expectation):** Assign each sample $\mathbf{x}_i$ to the nearest centroid:
   $$c_i^{(t)} = \arg\min_{k \in \{1, \dots, K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|_2^2$$
2. **Update Step (Maximization):** Recompute centroids:
   $$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{i: c_i = k} \mathbf{x}_i$$
Convergence is guaranteed because $J$ decreases monotonically at each iteration and the number of distinct partitions is finite ($K^n$).

#### K-Means++ Initialization
Standard random initialization risks poor local minima. K-Means++ ensures spread-out seeds:
1. Pick first centroid $\boldsymbol{\mu}_1$ uniformly at random from $\mathcal{X}$.
2. For each subsequent centroid $k = 2, \dots, K$, select $\mathbf{x} \in \mathcal{X}$ with probability proportional to the squared distance to its nearest already chosen centroid:
   $$P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}' \in \mathcal{X}} D(\mathbf{x}')^2}, \quad \text{where } D(\mathbf{x}) = \min_{j < k} \|\mathbf{x} - \boldsymbol{\mu}_j\|_2$$

### 1.2 Hierarchical Clustering & Linkage Metrics
Agglomerative (bottom-up) hierarchical clustering starts with $n$ singleton clusters and iteratively merges the closest pair according to a linkage criterion:
1. **Single Linkage (Minimum Distance):**
   $$d(A, B) = \min_{\mathbf{x} \in A, \mathbf{y} \in B} \|\mathbf{x} - \mathbf{y}\| \quad \implies \text{Susceptible to Chaining Effect (elongated clusters)}$$
2. **Complete Linkage (Maximum Distance):**
   $$d(A, B) = \max_{\mathbf{x} \in A, \mathbf{y} \in B} \|\mathbf{x} - \mathbf{y}\| \quad \implies \text{Produces compact, spherical clusters}$$
3. **Average Linkage (Mean Proximity):**
   $$d(A, B) = \frac{1}{|A| |B|} \sum_{\mathbf{x} \in A} \sum_{\mathbf{y} \in B} \|\mathbf{x} - \mathbf{y}\| \quad \implies \text{Robust to noise and outliers}$$
4. **Ward's Linkage:** Minimizes total within-cluster variance increase upon merging:
   $$\Delta \text{ESS}_{AB} = \frac{|A| |B|}{|A| + |B|} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|_2^2$$

---

## 2. Association Rule Mining

Given items $\mathcal{I} = \{i_1, \dots, i_m\}$ and transactions $\mathcal{T} = \{t_1, \dots, t_n\}$, an association rule is $X \implies Y$ where $X, Y \subset \mathcal{I}$ and $X \cap Y = \emptyset$.

### 2.1 Quantitative Evaluation Metrics
1. **Support:** Joint transaction probability:
   $$\text{Support}(X \implies Y) = P(X \cup Y) = \frac{|\{t \in \mathcal{T} : X \cup Y \subseteq t\}|}{|\mathcal{T}|}$$
2. **Confidence:** Conditional probability of $Y$ given $X$:
   $$\text{Confidence}(X \implies Y) = P(Y | X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$
3. **Lift:** Ratio of observed joint frequency to expected frequency under statistical independence:
   $$\text{Lift}(X \implies Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) \cdot \text{Support}(Y)}$$
   - $\text{Lift} = 1$: $X$ and $Y$ are independent.
   - $\text{Lift} > 1$: Positive association (complementary products).
   - $\text{Lift} < 1$: Negative association (substitute products).

### 2.2 Apriori vs FP-Growth
* **Apriori Algorithm:** Uses the **Downward Closure Property** (*Any subset of a frequent itemset must also be frequent; if an itemset is infrequent, all its supersets are infrequent*). Requires multiple full database scans $\mathcal{O}(k)$ and generates massive candidate sets $C_k$.
* **FP-Growth Algorithm:** Eliminates candidate generation entirely. Constructs a compact **Frequent Pattern Tree (FP-Tree)** using only 2 database passes and mines it recursively by building **Conditional Pattern Bases**.

---

## 3. Semi-Supervised Learning Algorithms

In real-world settings, labeled data $\mathcal{D}_L = \{(\mathbf{x}_i, y_i)\}_{i=1}^l$ is scarce and expensive, while unlabeled data $\mathcal{D}_U = \{\mathbf{x}_j\}_{j=l+1}^{l+u}$ is abundant ($u \gg l$).

### 3.1 Self-Training (Pseudo-Labeling)
1. Train supervised base classifier $f$ on $\mathcal{D}_L$.
2. Predict labels and confidence probabilities $\hat{p}_j = \max_c P(y=c|\mathbf{x}_j)$ on $\mathcal{D}_U$.
3. Select candidates with confidence above threshold: $\mathcal{S} = \{\mathbf{x}_j \in \mathcal{D}_U : \hat{p}_j \ge \tau\}$.
4. Augment labeled set: $\mathcal{D}_L \leftarrow \mathcal{D}_L \cup \{(\mathbf{x}_j, \hat{y}_j)\}_{\mathbf{x}_j \in \mathcal{S}}$, and remove them from $\mathcal{D}_U$.
5. Retrain classifier $f$ iteratively until convergence.
*Risk:* Early misclassifications reinforce confirmation bias.

### 3.2 Graph-Based Learning: Label Propagation vs Label Spreading
Construct an undirected graph where nodes are samples and edges represent affinities:
$$W_{ij} = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right) \quad (\text{with } W_{ii} = 0)$$

1. **Label Propagation Algorithm (LPA):**
   - Row-normalize transition matrix: $P = D^{-1} W$, where $D_{ii} = \sum_j W_{ij}$.
   - Iterate: $Y^{(t+1)} = P Y^{(t)}$.
   - **Hard Clamping:** Reset known labeled rows: $Y_L^{(t+1)} = Y_L^{(0)}$.
2. **Label Spreading:**
   - Uses symmetrically normalized Graph Laplacian: $S = D^{-1/2} W D^{-1/2}$.
   - **Soft Clamping:** Permits noise in initial labels using parameter $\alpha \in (0, 1)$:
     $$Y^{(t+1)} = \alpha S Y^{(t)} + (1 - \alpha) Y^{(0)}$$
   - Closed-form steady-state solution: $Y^* = (1 - \alpha)(I - \alpha S)^{-1} Y^{(0)}$.

---

## 4. Step-by-Step Solved 10-Mark Numerical Problem (VIT Format)

<div class="exam-box">
<strong>VIT Fall Semester Exam Question [10 Marks]:</strong><br>
Consider a 1-dimensional dataset $\mathcal{X} = \{2, 4, 10, 12, 3, 20, 30, 11, 25\}$.<br>
Initialize $K=2$ clusters with initial centroids $\mu_1 = 2$ and $\mu_2 = 4$.<br>
(a) Execute Iteration 1 of K-Means clustering. Determine cluster assignments and updated centroids. (4 Marks)<br>
(b) Execute Iteration 2. Determine if the algorithm converges. (3 Marks)<br>
(c) Compute the final Within-Cluster Sum of Squares (WCSS). (3 Marks)
</div>

#### Step 1: Iteration 1 Assignment
Distance to $\mu_1 = 2$ vs $\mu_2 = 4$:
- $x = 2$: $|2-2|=0 \le |2-4|=2 \implies C_1$
- $x = 4$: $|4-2|=2 > |4-4|=0 \implies C_2$
- $x = 10$: $|10-2|=8 > |10-4|=6 \implies C_2$
- $x = 12$: $|12-2|=10 > |12-4|=8 \implies C_2$
- $x = 3$: $|3-2|=1 \le |3-4|=1 \implies C_1$ (tie broken to $C_1$)
- $x = 20, 30, 11, 25$: All clearly closer to $\mu_2=4 \implies C_2$

Partition 1:
- $C_1 = \{2, 3\}$
- $C_2 = \{4, 10, 11, 12, 20, 25, 30\}$

#### Step 2: Iteration 1 Centroid Update
$$\mu_1^{(1)} = \frac{2 + 3}{2} = \mathbf{2.5}$$
$$\mu_2^{(1)} = \frac{4 + 10 + 11 + 12 + 20 + 25 + 30}{7} = \frac{112}{7} = \mathbf{16.0}$$

#### Step 3: Iteration 2 Assignment
Midpoint threshold between $\mu_1 = 2.5$ and $\mu_2 = 16.0$ is $\frac{2.5 + 16.0}{2} = 9.25$.
- Samples $< 9.25$: $\{2, 3, 4\} \implies C_1$
- Samples $> 9.25$: $\{10, 11, 12, 20, 25, 30\} \implies C_2$

Recompute Centroids for Iteration 2:
$$\mu_1^{(2)} = \frac{2 + 3 + 4}{3} = \frac{9}{3} = \mathbf{3.0}$$
$$\mu_2^{(2)} = \frac{10 + 11 + 12 + 20 + 25 + 30}{6} = \frac{108}{6} = \mathbf{18.0}$$

Check Iteration 3 Assignment:
Midpoint between 3.0 and 18.0 is $\frac{3.0 + 18.0}{2} = 10.5$.
- $x = 10 < 10.5 \implies$ switches to $C_1$!
- Partition: $C_1 = \{2, 3, 4, 10\}$, $C_2 = \{11, 12, 20, 25, 30\}$.
New Centroids:
$$\mu_1^{(3)} = \frac{2 + 3 + 4 + 10}{4} = \frac{19}{4} = \mathbf{4.75}$$
$$\mu_2^{(3)} = \frac{11 + 12 + 20 + 25 + 30}{5} = \frac{98}{5} = \mathbf{19.6}$$
Midpoint between 4.75 and 19.6 is $12.175$. Now $11, 12 < 12.175$, so they switch to $C_1$ until convergence at $C_1 = \{2, 3, 4, 10, 11, 12\}$ and $C_2 = \{20, 25, 30\}$.
Final Centroids: $\mu_1 = 7.0$, $\mu_2 = 25.0$.

#### Step 4: Final WCSS
- For $C_1$: $(2-7)^2 + (3-7)^2 + (4-7)^2 + (10-7)^2 + (11-7)^2 + (12-7)^2 = 25 + 16 + 9 + 9 + 16 + 25 = 100$
- For $C_2$: $(20-25)^2 + (25-25)^2 + (30-25)^2 = 25 + 0 + 25 = 50$
$$\text{WCSS} = 100 + 50 = \mathbf{150.0}$$
