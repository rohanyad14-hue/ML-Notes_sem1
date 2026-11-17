# MACSE504: Machine Learning - Module 1 | Topic 1
# Logistic Regression, Binary Cross-Entropy Loss & Multi-Class Classification
**Course:** MACSE504 Machine Learning | **Credits:** 4 | **Target:** 10-Mark VIT Examination Questions  
**Prescribed Textbooks:**  
1. L. Goel, *Artificial Intelligence: Concepts and Applications*, Wiley, 2021.  
2. Andreas Müller & Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2016.  
3. P. Dangeti, *Statistics for Machine Learning*, Packt Publishing, 2017.

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **Krish Naik - Logistic Regression Geometric Intuition & Math (18 min):** [Watch on YouTube](https://www.youtube.com/watch?v=L_x_fGz_Vuo)  
  *Core Focus:* Sigmoid activation function, odds ratio, and linear decision boundary.
* **CampusX (Nitish Singh) - Logistic Regression Cost Function & Gradient Descent (19 min):** [Watch on YouTube](https://www.youtube.com/watch?v=uF_8n2GvU0I)  
  *Core Focus:* Derivation of binary cross-entropy log-loss and parameter gradient updates.
* **Gate Smashers (Varun Singla) - Logistic Regression in Machine Learning (12 min):** [Watch on YouTube](https://www.youtube.com/watch?v=zAULhXNZ3r4)  
  *Core Focus:* Linear vs. logistic classification, probability thresholds, and multiclass OvR.

---

## 1. Mathematical Formulation of Logistic Regression

Logistic Regression is a supervised discriminative classification algorithm that models the posterior probability of the positive class $P(Y=1|X)$ using the **Logistic / Sigmoid Function**.

### 1.1 The Logit Link & Sigmoid Function
Let linear predictor $z = w^T x + b = \sum_{j=1}^p w_j x_j + b$.
$$\sigma(z) = rac{1}{1 + e^{-z}} = rac{e^z}{1 + e^z}$$
* **Range:** $\sigma(z) \in (0, 1)$, allowing output interpretation as a valid probability.
* **Odds & Log-Odds (Logit):**
  $$	ext{Odds} = rac{P(Y=1|x)}{1 - P(Y=1|x)} = e^{w^T x + b} \implies \ln\left( rac{P}{1 - P} ight) = w^T x + b$$

---

## 2. Binary Cross-Entropy (Log-Loss) Cost Function

Linear regression's Mean Squared Error (MSE) is non-convex when combined with the sigmoid function, leading to local minima. Logistic regression utilizes the **Maximum Likelihood Estimation (MLE)** derived **Binary Cross-Entropy Loss**:

$$J(w, b) = -rac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \Big]$$
where $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$.

### 2.1 Gradient Descent Parameter Updates
$$rac{\partial J}{\partial w_j} = rac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$
$$rac{\partial J}{\partial b} = rac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$
$$w_j \leftarrow w_j - lpha rac{\partial J}{\partial w_j}, \quad b \leftarrow b - lpha rac{\partial J}{\partial b}$$

---

## 3. Multi-Class Classification Strategies

1. **One-vs-Rest (OvR / One-vs-All):** Train $K$ independent binary classifiers, each distinguishing class $k$ from all other $K-1$ classes combined. Predict class with highest confidence score: $\hat{y} = rg\max_k \hat{P}_k(x)$.
2. **One-vs-One (OvO):** Train $rac{K(K-1)}{2}$ binary classifiers for all pairwise class combinations. Predict via majority voting.
3. **Multinomial Logistic Regression (Softmax Regression):** Generalizes sigmoid to $K$ classes:
   $$P(Y = k | x) = rac{e^{w_k^T x}}{\sum_{j=1}^K e^{w_j^T x}}$$

---

## 4. Model 10-Mark Solved Written Problems for VIT Examination

<div class="box-example">
<h4>10-Mark Solved Problem 1.1: Logistic Regression Probability & Log-Loss Calculation</h4>
<strong>Question:</strong> Given a binary classification problem with weights $w = [0.5, -0.8]$, bias $b = 0.2$, and learning rate $lpha = 0.1$.
<br>(a) For input sample $x = [2.0, 1.5]$ with true label $y = 1$, compute the predicted probability $\hat{y}$.
<br>(b) Calculate the Log-Loss for this individual sample.
<br>(c) Perform one step of Gradient Descent update for weights $w_1, w_2$ and bias $b$. (10 Marks)
<br><br>
<strong>(a) Calculate Predicted Probability $\hat{y}$:</strong>
$$z = w_1 x_1 + w_2 x_2 + b = (0.5)(2.0) + (-0.8)(1.5) + 0.2 = 1.0 - 1.2 + 0.2 = 0.0$$
$$\hat{y} = \sigma(z) = rac{1}{1 + e^{-0}} = rac{1}{1 + 1} = \mathbf{0.500}$$
<br>
<strong>(b) Calculate Binary Log-Loss:</strong>
$$L = -[y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y})] = -[1 \cdot \ln(0.5) + 0] = -\ln(0.5) = \ln(2) pprox \mathbf{0.6931}$$
<br>
<strong>(c) Gradient Descent Parameter Updates:</strong>
Error term: $e = \hat{y} - y = 0.5 - 1.0 = -0.5$.
- $rac{\partial J}{\partial w_1} = e \cdot x_1 = (-0.5)(2.0) = -1.0$
- $rac{\partial J}{\partial w_2} = e \cdot x_2 = (-0.5)(1.5) = -0.75$
- $rac{\partial J}{\partial b} = e = -0.5$
<br>
Updates with $lpha = 0.1$:
$$w_1 \leftarrow w_1 - lpha(-1.0) = 0.5 - (0.1)(-1.0) = 0.5 + 0.1 = \mathbf{0.60}$$
$$w_2 \leftarrow w_2 - lpha(-0.75) = -0.8 - (0.1)(-0.75) = -0.8 + 0.075 = \mathbf{-0.725}$$
$$b \leftarrow b - lpha(-0.5) = 0.2 - (0.1)(-0.5) = 0.2 + 0.05 = \mathbf{0.25}$$
</div>
