# MACSE504: Machine Learning - Module 1 | Topic 2
# K-Nearest Neighbors (KNN), Distance Metrics & The Curse of Dimensionality
**Course:** MACSE504 Machine Learning | **Credits:** 4 | **Target:** 10-Mark VIT Examination Questions  
**Prescribed Textbooks:**  
1. Andreas Müller & Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2016 (Chapter 2).  
2. L. Goel, *Artificial Intelligence: Concepts and Applications*, Wiley, 2021.  
3. T. Jo, *Machine Learning Foundations*, Springer Nature, 2021.

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **CampusX (Nitish Singh) - KNN Algorithm In-Depth with Distance Metrics (18 min):** [Watch on YouTube](https://www.youtube.com/watch?v=4HKqjENfl9U)  
  *Core Focus:* Geometric intuition, Euclidean/Manhattan metrics, and boundary behavior.
* **Gate Smashers (Varun Singla) - K-Nearest Neighbor Algorithm with Solved Example (13 min):** [Watch on YouTube](https://www.youtube.com/watch?v=HVXime0nQeI)  
  *Core Focus:* Choosing optimal $K$, tie breaking, and voting mechanics.
* **Krish Naik - Curse of Dimensionality in KNN (14 min):** [Watch on YouTube](https://www.youtube.com/watch?v=QZ0vtL4z10E)  
  *Core Focus:* Geometric distance concentration and why KNN fails in high dimensions.

---

## 1. Principles of K-Nearest Neighbors (KNN)

KNN is a non-parametric, instance-based (lazy learning) algorithm:
* **Lazy Learning:** KNN stores the complete training dataset during `fit()` ($O(1)$ training time) and postpones all computation until query time ($O(N \cdot D)$ inference time).
* **Inductive Bias:** Assumes that instances close to each other in feature space share identical class labels or target values.

### 1.1 Distance Metrics
For vectors $x = [x_1, \dots, x_p]^T$ and $z = [z_1, \dots, z_p]^T$:
1. **Euclidean Distance ($L_2$ Norm):** $d_2(x, z) = \sqrt{\sum_{j=1}^p (x_j - z_j)^2}$
2. **Manhattan Distance ($L_1$ Norm):** $d_1(x, z) = \sum_{j=1}^p |x_j - z_j|$
3. **Minkowski Distance ($L_p$ Metric):** $d_p(x, z) = \left( \sum_{j=1}^p |x_j - z_j|^p ight)^{1/p}$
4. **Cosine Similarity (Text / High Dimension):** $	ext{Cosine}(x, z) = rac{x \cdot z}{\|x\| \|z\|}$

---

## 2. Selecting Hyperparameter $K$ & Curse of Dimensionality

* **$K = 1$ (Extreme Overfitting):** High variance, zero training error, highly sensitive to noise and outliers.
* **Large $K pprox N$ (Extreme Underfitting):** High bias, predicts global majority class everywhere.
* **Rule of Thumb:** $K = \sqrt{N}$ (choose odd $K$ for binary classification to avoid tie-breaks).
* **Curse of Dimensionality:** In high-dimensional spaces ($p 	o \infty$), the volume of space grows exponentially ($V \propto r^p$). The distance to the nearest neighbor converges toward the distance to the farthest neighbor:
  $$\lim_{p 	o \infty} rac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
  All points become equidistant, rendering proximity metrics meaningless.

---

## 3. Model 10-Mark Solved Problems for VIT Examination

<div class="box-example">
<h4>10-Mark Solved Problem 2.1: Step-by-Step KNN Numerical Classification</h4>
<strong>Question:</strong> Given the following 2D training dataset with labels:
<table border="1" cellpadding="5">
<tr><th>Instance</th><th>Feature $X_1$</th><th>Feature $X_2$</th><th>Class Label</th></tr>
<tr><td>A</td><td>1.0</td><td>2.0</td><td>Positive (+)</td></tr>
<tr><td>B</td><td>2.0</td><td>1.0</td><td>Positive (+)</td></tr>
<tr><td>C</td><td>2.0</td><td>3.0</td><td>Positive (+)</td></tr>
<tr><td>D</td><td>6.0</td><td>5.0</td><td>Negative (-)</td></tr>
<tr><td>E</td><td>7.0</td><td>6.0</td><td>Negative (-)</td></tr>
<tr><td>F</td><td>6.0</td><td>7.0</td><td>Negative (-)</td></tr>
</table>
A new query point $Q = (3.0, 4.0)$ arrives.
<br>(a) Compute Euclidean distances from $Q$ to all training points.
<br>(b) Predict class label for $K = 3$ (Majority Voting).
<br>(c) Predict class label using Distance-Weighted KNN ($w_i = 1 / d_i^2$). (10 Marks)
<br><br>
<strong>(a) Compute Euclidean Distances:</strong>
- $d(Q, A) = \sqrt{(3-1)^2 + (4-2)^2} = \sqrt{4 + 4} = \sqrt{8} pprox \mathbf{2.828}$
- $d(Q, B) = \sqrt{(3-2)^2 + (4-1)^2} = \sqrt{1 + 9} = \sqrt{10} pprox \mathbf{3.162}$
- $d(Q, C) = \sqrt{(3-2)^2 + (4-3)^2} = \sqrt{1 + 1} = \sqrt{2} pprox \mathbf{1.414}$
- $d(Q, D) = \sqrt{(3-6)^2 + (4-5)^2} = \sqrt{9 + 1} = \sqrt{10} pprox \mathbf{3.162}$
- $d(Q, E) = \sqrt{(3-7)^2 + (4-6)^2} = \sqrt{16 + 4} = \sqrt{20} pprox \mathbf{4.472}$
- $d(Q, F) = \sqrt{(3-6)^2 + (4-7)^2} = \sqrt{9 + 9} = \sqrt{18} pprox \mathbf{4.243}$
<br>
<strong>Sorted Neighbors:</strong>
1. Point C: $d = 1.414$ [Class +]
2. Point A: $d = 2.828$ [Class +]
3. Point B: $d = 3.162$ [Class +]
4. Point D: $d = 3.162$ [Class -]
<br>
<strong>(b) Majority Voting for $K = 3$:</strong>
The 3 nearest neighbors are C (+), A (+), and B (+).
All 3 votes belong to Positive (+).
$$\hat{y}_{K=3} = \mathbf{	ext{Positive (+)}}$$
<br>
<strong>(c) Distance-Weighted Voting for $K = 3$:</strong>
- Weight for C: $w_C = 1 / (\sqrt{2})^2 = 1/2 = 0.50$
- Weight for A: $w_A = 1 / (\sqrt{8})^2 = 1/8 = 0.125$
- Weight for B: $w_B = 1 / (\sqrt{10})^2 = 1/10 = 0.10$
<br>
Total Weighted Vote for (+) $= 0.50 + 0.125 + 0.10 = \mathbf{0.725}$. Vote for (-) $= 0.0$.
$$\hat{y}_{	ext{weighted}} = \mathbf{	ext{Positive (+)}}$$
</div>
