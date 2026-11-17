# MACSE504: Machine Learning - Module 1 | Topic 4
# Decision Trees: Impurity Criteria (Entropy, Information Gain, Gini) & Pruning
**Course:** MACSE504 Machine Learning | **Credits:** 4 | **Target:** 10-Mark VIT Examination Questions  
**Prescribed Textbooks:**  
1. Andreas Müller & Sarah Guido, *Introduction to Machine Learning with Python*, O'Reilly, 2016.  
2. L. Goel, *Artificial Intelligence: Concepts and Applications*, Wiley, 2021.  
3. P. Dangeti, *Statistics for Machine Learning*, Packt Publishing, 2017.

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **CampusX (Nitish Singh) - Decision Tree Math: Entropy, Information Gain & Gini (19 min):** [Watch on YouTube](https://www.youtube.com/watch?v=1TQ_eF1F27E)  
  *Core Focus:* Derivation of Shannon entropy, reduction calculation, and ID3 vs. CART.
* **Gate Smashers (Varun Singla) - Decision Tree with Solved Numerical (15 min):** [Watch on YouTube](https://www.youtube.com/watch?v=coOTEc-0OGw)  
  *Core Focus:* Full step-by-step Information Gain table trace for exam questions.
* **Krish Naik - Decision Tree Pre-Pruning and Post-Pruning (14 min):** [Watch on YouTube](https://www.youtube.com/watch?v=SLOyyFHhNps)  
  *Core Focus:* Cost-complexity pruning $\alpha$ and avoiding tree overfitting.

---

## 1. Splitting Criteria Formulations

<div class="diagram-box">
  <img src="images/ml_decision_tree.svg" style="max-width:540px;" alt="Decision Tree Splitting">
  <div class="diagram-caption">Figure 4.1: Decision Tree Architecture with recursive binary splitting, Gini Impurity, and leaf classifications.</div>
</div>

### 1.1 Shannon Entropy & Information Gain (ID3 Algorithm)
* **Entropy:** Expected information content / uncertainty of set $S$:
  $$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$
* **Information Gain (IG):** Expected reduction in entropy from splitting on attribute $A$:
  $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

### 1.2 Gini Impurity (CART Algorithm)
Probability of incorrectly labeling a randomly chosen element from set $S$:
$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$
* **Pure Node:** $Gini = 0.0$. Balanced 2-class node: $Gini = 1 - (0.5^2 + 0.5^2) = 0.50$.
* Computationally faster than Entropy because it requires no logarithmic evaluations.

---

## 2. Pruning Strategies to Combat Overfitting

* **Pre-Pruning (Early Stopping):** Stop splitting when tree reaches `max_depth`, `min_samples_split`, or when $IG < \epsilon$.
* **Post-Pruning (Cost-Complexity Pruning):** Grow a full unconstrained tree $T_0$, then minimize cost-complexity objective:
  $$R_\alpha(T) = R(T) + \alpha |T|$$
  where $R(T)$ is classification error rate, $|T|$ is number of terminal leaves, and $\alpha$ is complexity tuning parameter.

---

## 3. Model 10-Mark Solved Problems for VIT Examination

<div class="box-example">
<h4>10-Mark Solved Problem 4.1: Full Information Gain & Root Node Split Calculation</h4>
<strong>Question:</strong> A cancer diagnostic dataset contains 14 patients. Target variable is Tumor Diagnosis: 9 Malignant (M) and 5 Benign (B).
<br>Consider attribute <strong>Texture</strong> with values: <em>Smooth</em> (4 M, 2 B) and <em>Rough</em> (5 M, 3 B).
<br>(a) Calculate the Entropy of the entire dataset $H(S)$.
<br>(b) Calculate the conditional entropy for Texture $H(S|\text{Texture})$.
<br>(c) Compute Information Gain $IG(S, \text{Texture})$. (10 Marks - Standard VIT Lab 1 Pattern)
<br><br>
<strong>(a) Calculate Dataset Entropy $H(S)$:</strong>
Total $N = 14$. $p_M = 9/14 \approx 0.6429$, $p_B = 5/14 \approx 0.3571$.
$$H(S) = -\left[ \frac{9}{14} \log_2\left(\frac{9}{14}\right) + \frac{5}{14} \log_2\left(\frac{5}{14}\right) \right]$$
$$= -[0.6429(-0.6374) + 0.3571(-1.4854)] = -[-0.4098 - 0.5304] = \mathbf{0.9403 \text{ bits}}$$
<br>
<strong>(b) Calculate Conditional Entropy:</strong>
1. <strong>Subset <em>Smooth</em> ($N = 6$):</strong> 4 M, 2 B.
   $$H(S_{\text{Smooth}}) = -\left[ \frac{4}{6} \log_2\left(\frac{4}{6}\right) + \frac{2}{6} \log_2\left(\frac{2}{6}\right) \right] = -[0.6667(-0.585) + 0.3333(-1.585)] = 0.390 + 0.528 = \mathbf{0.9183 \text{ bits}}$$
2. <strong>Subset <em>Rough</em> ($N = 8$):</strong> 5 M, 3 B.
   $$H(S_{\text{Rough}}) = -\left[ \frac{5}{8} \log_2\left(\frac{5}{8}\right) + \frac{3}{8} \log_2\left(\frac{3}{8}\right) \right] = -[0.625(-0.678) + 0.375(-1.415)] = 0.4238 + 0.5306 = \mathbf{0.9544 \text{ bits}}$$
<br>
Weighted Conditional Entropy:
$$H(S | \text{Texture}) = \frac{6}{14} H(S_{\text{Smooth}}) + \frac{8}{14} H(S_{\text{Rough}})$$
$$= \frac{6}{14}(0.9183) + \frac{8}{14}(0.9544) = 0.3936 + 0.5454 = \mathbf{0.9390 \text{ bits}}$$
<br>
<strong>(c) Compute Information Gain:</strong>
$$IG(S, \text{Texture}) = H(S) - H(S | \text{Texture}) = 0.9403 - 0.9390 = \mathbf{0.0013 \text{ bits}}$$
<em>Conclusion:</em> Texture yields minimal Information Gain ($0.0013$), meaning it provides negligible discriminatory power on this dataset.
</div>
