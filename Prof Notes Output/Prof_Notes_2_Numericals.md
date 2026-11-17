# MACSE504 Machine Learning: NUMERICALS (Worked Examples)

<p class="cover"><b>N1-N18:</b> every worked example in <b>Prof Provided Notes</b>. <b>N19-N29 (Part B):</b> practice problems I built for syllabus topics that folder does not cover. All are redone step by step and <b>re-computed in Python to confirm the answers</b>. Format for each: <b>Given → Method → Working → Answer</b>. Where the professor's file has an arithmetic slip, it is marked <b>⚠ Check</b> with the corrected value.<br>Theory is in <b>Prof_Notes_1_Theory</b>; symbols are explained in <b>Prof_Notes_3_Formula_Sheet</b>.</p>

**Index:** N1 Simple Linear Regression · N2 Multiple Linear Regression · N3 Decision Tree (Info Gain/Gini) · N4 Naïve Bayes · N5 SVM (linear, non-linear, kernel) · N6 SVR · N7 K-means · N8 Hierarchical (single/complete link) · N9 Apriori & FP-Growth · N10 Classification & Regression Metrics · N11 Imbalanced Data · N12 Self-training · N13 Label Propagation · N14 Label Spreading · N15 GNN message passing · N16 PCA (two examples) · N17 LLE · N18 Reinforcement Learning (6 examples)

---

## N1. Simple Linear Regression

### N1a. Estriol vs birth-weight (slides 8-10)
**Given** $(x,y)$: (1,1), (2,1), (3,2), (4,2), (5,4); $n=5$.

| $x_i$ | $y_i$ | $x_i^2$ | $y_i^2$ | $x_iy_i$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 1 | 4 | 1 | 2 |
| 3 | 2 | 9 | 4 | 6 |
| 4 | 2 | 16 | 4 | 8 |
| 5 | 4 | 25 | 16 | 20 |
| **15** | **10** | **55** | **26** | **37** |

$$\hat\beta_1=\frac{\sum xy-\frac{(\sum x)(\sum y)}{n}}{\sum x^2-\frac{(\sum x)^2}{n}}=\frac{37-\frac{15\cdot10}{5}}{55-\frac{15^2}{5}}=\frac{37-30}{55-45}=\mathbf{0.70}$$
$$\bar X=3,\ \bar Y=2,\qquad \hat\beta_0=\bar Y-\hat\beta_1\bar X=2-0.70(3)=\mathbf{-0.10}$$
**Answer:** $\hat y=-0.10+0.70x$.

### N1b. Pizza sales (slide 11)
**Given:** (2,58), (6,105), (8,88), (8,118), (12,117). Predict sales for $X=10$ (10 000 students).

$\bar x=7.2,\ \bar y=97.2,\ S_{xy}=298.8,\ S_{xx}=52.8$ → $\beta_1=298.8/52.8=5.659$, $\beta_0=97.2-5.659(7.2)=56.45$, so $\hat y(10)=56.45+56.59=113.05$.

> ⚠ **Check:** the slide states $\beta_0=60,\ \beta_1=5,\ \hat Y=60+5X$ and **$\hat Y(10)=110$ ⇒ \$110 000**. Those coefficients are the answer for the *full 10-point* textbook dataset; the five points shown give $56.45+5.66X$. **If asked in an exam, use the professor's $\hat Y=60+5X$ and quote 110 (\$110 000)**; just know the five listed points alone do not reproduce it.

### N1c. Exercises (slide 16)
1. $\hat Y=15+0.12X$, $X=500$ kWh → $\hat Y=15+0.12(500)=15+60=\mathbf{\$75}$.
2. $\beta_1=-3.5$: **negative (inverse) relationship**. For each 1-unit increase in the predictor, the target is expected to **decrease by 3.5 units** on average.

---

## N2. Multiple Linear Regression by hand (Statology PDF)

**Given** ($n=8$): 

| y | 140 | 155 | 159 | 179 | 192 | 200 | 212 | 215 |
|---|---|---|---|---|---|---|---|---|
| $x_1$ | 60 | 62 | 67 | 70 | 71 | 72 | 75 | 78 |
| $x_2$ | 22 | 25 | 24 | 20 | 15 | 14 | 14 | 11 |

**Step 1: column sums:** $\sum y=1452,\ \sum x_1=555,\ \sum x_2=145$; $\sum x_1^2=38767,\ \sum x_2^2=2823,\ \sum x_1y=101895,\ \sum x_2y=25364,\ \sum x_1x_2=9859$. Means: $\bar y=181.5,\ \bar x_1=69.375,\ \bar x_2=18.125$.

**Step 2: regression sums** (corrected sums, $\sum x_a x_b=\sum X_aX_b-\frac{\sum X_a\sum X_b}{n}$):

| $\sum x_1^2$ | $\sum x_2^2$ | $\sum x_1y$ | $\sum x_2y$ | $\sum x_1x_2$ |
|---|---|---|---|---|
| $38767-\frac{555^2}{8}=263.875$ | $2823-\frac{145^2}{8}=194.875$ | $101895-\frac{555\cdot1452}{8}=1162.5$ | $25364-\frac{145\cdot1452}{8}=-953.5$ | $9859-\frac{555\cdot145}{8}=-200.375$ |

**Step 3: coefficients**
$$b_1=\frac{(\sum x_2^2)(\sum x_1y)-(\sum x_1x_2)(\sum x_2y)}{(\sum x_1^2)(\sum x_2^2)-(\sum x_1x_2)^2}=\frac{(194.875)(1162.5)-(-200.375)(-953.5)}{(263.875)(194.875)-(-200.375)^2}=\mathbf{3.148}$$
$$b_2=\frac{(\sum x_1^2)(\sum x_2y)-(\sum x_1x_2)(\sum x_1y)}{(\sum x_1^2)(\sum x_2^2)-(\sum x_1x_2)^2}=\mathbf{-1.656},\qquad b_0=\bar y-b_1\bar x_1-b_2\bar x_2=181.5-3.148(69.375)+1.656(18.125)=\mathbf{-6.867}$$

**Answer:** $\hat y=-6.867+3.148x_1-1.656x_2$.
*Interpretation:* $b_1$: +1 in $x_1$ → +3.148 in $y$ (holding $x_2$ fixed). $b_2$: +1 in $x_2$ → −1.656 in $y$ (holding $x_1$ fixed). $b_0$: mean $y$ when both predictors are 0.

> ⚠ **Check:** the PDF's $b_2$ line prints "1152.5" but should be **1162.5** ($\sum x_1y$); the final −1.656 is correct. The PDF also skips "Step 4".

---

## N3. Decision Tree: Information Gain, Gain Ratio, Gini (AllElectronics, 14 tuples)

**Given:** `buys_computer` = 9 yes, 5 no. Attributes: age, income, student, credit_rating.

**Base entropy**
$$\text{Info}(D)=-\tfrac{9}{14}\log_2\tfrac{9}{14}-\tfrac{5}{14}\log_2\tfrac{5}{14}=0.940\text{ bits}$$

**Split on age** (youth 2 yes/3 no, middle-aged 4/0, senior 3/2):
$$\text{Info}_{age}(D)=\tfrac{5}{14}\underbrace{(0.971)}_{\text{Info}(2,3)}+\tfrac{4}{14}(0)+\tfrac{5}{14}(0.971)=0.694$$
$$\text{Gain}(age)=0.940-0.694=\mathbf{0.246}$$

| Attribute | Gain | |
|---|---|---|
| **age** | **0.246** | ← highest → **root** |
| student | 0.151 | |
| credit_rating | 0.048 | |
| income | 0.029 | |

* **Middle-aged** branch has 4 yes / 0 no → **pure → leaf "yes"** immediately. Youth and senior are impure → split again using student / credit_rating / income.
* **Gain ratio for age (extra, computed):** $\text{SplitInfo}_{age}=-\tfrac5{14}\log_2\tfrac5{14}-\tfrac4{14}\log_2\tfrac4{14}-\tfrac5{14}\log_2\tfrac5{14}=1.577$; $\text{GainRatio}=0.246/1.577=0.156$.
* **Gini of the whole set (extra):** $1-(9/14)^2-(5/14)^2=1-0.413-0.128=0.459$.
* **AVC-set (RainForest slide):** Youth 30 buy / 50 no-buy; Middle_Age 80 / 0; Senior 40 / 20.

**Colab (Iris, 70/30 split, depth 3, min_samples_leaf 5):** both Gini and Entropy trees give confusion matrix $\begin{bmatrix}16&0&0\\0&10&1\\0&1&17\end{bmatrix}$; accuracy $=(16+10+17)/45=43/45=\mathbf{95.56\%}$; class precisions/recalls 1.00, 0.91, 0.94.

---

## N4. Naïve Bayes

### N4a. Play-Tennis (discrete)
**Query:** $x'=$ (Outlook=Sunny, Temp=Cool, Humidity=High, Wind=Strong). **Priors:** $P(\text{Yes})=9/14$, $P(\text{No})=5/14$.

| Feature | $P(\cdot\mid\text{Yes})$ | $P(\cdot\mid\text{No})$ |
|---|---|---|
| Sunny | 2/9 | 3/5 |
| Cool | 3/9 | 1/5 |
| High | 3/9 | 4/5 |
| Strong | 3/9 | 3/5 |

$$P(\text{Yes}\mid x')\propto\tfrac29\cdot\tfrac39\cdot\tfrac39\cdot\tfrac39\cdot\tfrac9{14}=\mathbf{0.0053},\qquad P(\text{No}\mid x')\propto\tfrac35\cdot\tfrac15\cdot\tfrac45\cdot\tfrac35\cdot\tfrac5{14}=\mathbf{0.0206}$$
$0.0053<0.0206$ ⇒ **predict "No"**. (Normalised: Yes $=0.0053/0.0259=0.205$, No $=0.795$.)

### N4b. Continuous feature (temperature)
Yes: 25.2, 19.3, 18.5, 21.7, 20.1, 24.3, 22.8, 23.1, 19.8 → $\mu_{Yes}=21.64,\ \sigma_{Yes}=2.35$.
No: 27.3, 30.1, 17.4, 29.5, 15.1 → $\mu_{No}=23.88,\ \sigma_{No}=7.09$.
$$\hat P(x\mid\text{Yes})=\frac{1}{2.35\sqrt{2\pi}}\exp\!\Big(-\frac{(x-21.64)^2}{2\times2.35^2}\Big),\quad \hat P(x\mid\text{No})=\frac{1}{7.09\sqrt{2\pi}}\exp\!\Big(-\frac{(x-23.88)^2}{2\times7.09^2}\Big)$$

> ⚠ **Check:** the slide writes $\sigma^2=\frac1N\sum(x-\mu)^2$, but its numbers (2.35, 7.09) are the **sample** std ($N-1$ divisor). With $1/N$ you would get 2.22 and 6.34. Use the slide's values in an exam.

### N4c. Zero probability (m-estimate)
$P(\text{overcast}\mid\text{no})=0$. "No" has $n=5$ examples; add $m=1$ virtual example; outlook has 3 values ⇒ $p=1/3$.
$$P(\text{overcast}\mid no)=\frac{0+1(1/3)}{5+1}=\frac1{18},\quad P(\text{sunny}\mid no)=\frac{3+1/3}{6}=\frac59,\quad P(\text{rain}\mid no)=\frac{2+1/3}{6}=\frac7{18}$$
(They still sum to 1: $1/18+10/18+7/18=1$.)

### N4d. Probability quiz (two dice)
$A$: die 1 = 3; $B$: die 2 = 1; $C$: sum = 8.
$P(A)=1/6$, $P(B)=1/6$, $P(C)=5/36$ [(2,6),(3,5),(4,4),(5,3),(6,2)]; $P(A\mid B)=\frac{1/36}{1/6}=1/6$; $P(C\mid A)=1/6$ (die 2 must be 5); $P(A,B)=1/36$; $P(A,C)=1/36$. $A,C$ are **not independent**: $P(A)P(C)=5/216\ne1/36$.

---

## N5. Support Vector Machines

### N5a. Linear SVM in $\mathbb R^2$ (Ventura)
**Positive:** (3,1), (3,−1), (6,1), (6,−1). **Negative:** (1,0), (0,1), (0,−1), (−1,0). By inspection, **support vectors** $s_1=(1,0)$ [−], $s_2=(3,1)$ [+], $s_3=(3,-1)$ [+]. Augment with bias input 1: $\tilde s_1=(1,0,1),\ \tilde s_2=(3,1,1),\ \tilde s_3=(3,-1,1)$.

**Dot products:** $\tilde s_1\!\cdot\!\tilde s_1=2$; $\tilde s_1\!\cdot\!\tilde s_2=\tilde s_1\!\cdot\!\tilde s_3=4$; $\tilde s_2\!\cdot\!\tilde s_2=\tilde s_3\!\cdot\!\tilde s_3=11$; $\tilde s_2\!\cdot\!\tilde s_3=9$.

$$\begin{aligned}2\alpha_1+4\alpha_2+4\alpha_3&=-1\\4\alpha_1+11\alpha_2+9\alpha_3&=+1\\4\alpha_1+9\alpha_2+11\alpha_3&=+1\end{aligned}\Rightarrow\ \alpha_1=-3.5,\ \alpha_2=\alpha_3=0.75$$
$$\tilde w=\sum\alpha_i\tilde s_i=-3.5(1,0,1)+0.75(3,1,1)+0.75(3,-1,1)=(1,\,0,\,-2)$$
**Answer:** $w=(1,0)$, $b=-2$ ⇒ boundary $x_1-2=0$, i.e. the vertical line $x=2$. (Last entry of $\tilde w$ is the bias.)

### N5b. Non-linear data with mapping $\Phi_1$
**Positive:** (2,2), (2,−2), (−2,−2), (−2,2). **Negative:** (1,1), (1,−1), (−1,−1), (−1,1). Not linearly separable in input space.
$$\Phi_1(x_1,x_2)=\begin{cases}(4-x_2+\lvert x_1-x_2\rvert,\ 4-x_1+\lvert x_1-x_2\rvert)&\sqrt{x_1^2+x_2^2}>2\\(x_1,x_2)&\text{otherwise}\end{cases}$$
Feature space: positives → (2,2), (6,2), (6,6), (2,6); negatives unchanged. **SVs:** $s_1=(1,1)$ [−], $s_2=(2,2)$ [+] ⇒ $\tilde s_1=(1,1,1),\ \tilde s_2=(2,2,1)$.
$$3\alpha_1+5\alpha_2=-1,\quad 5\alpha_1+9\alpha_2=+1\ \Rightarrow\ \alpha_1=-7,\ \alpha_2=4$$
$$\tilde w=-7(1,1,1)+4(2,2,1)=(1,1,-3)\ \Rightarrow\ w=(1,1),\ b=-3$$
**Classify $x=(4,5)$:** $\sqrt{41}>2$ → $\Phi_1=(4-5+1,\ 4-4+1)=(0,1)$, augmented $(0,1,1)$.
$f=\sigma\big(-7(1,1,1)\!\cdot\!(0,1,1)+4(2,2,1)\!\cdot\!(0,1,1)\big)=\sigma(-14+12)=\sigma(-2)$ ⇒ **negative class.**

### N5c. Higher-dimensional map $\Phi_2$ (kernel-trick motivation)
$\Phi_2(x_1,x_2)=\big(x_1,\ x_2,\ \tfrac{x_1^2+x_2^2-5}{3}\big)$: positives → $(\pm2,\pm2,\,1)$, negatives → $(\pm1,\pm1,\,-1)$. All 8 points are SVs with $\alpha=\tfrac1{46}$ (positives), $\alpha=-\tfrac7{46}$ (negatives). $w=(0,0,1)$, $b=0$ ⇒ **$f(x)=\sigma(x_3)$**: the third feature alone separates the classes (no augmentation needed since the hyperplane passes through the origin).

### N5d. 1-D data with polynomial kernel (slides 35-39)
**Given:** $x=\{1,2,4,5,6\}$, $y=\{+1,+1,-1,-1,+1\}$; kernel $K(x,y)=(xy+1)^2$; $C=100$.
Dual: maximise $\sum\alpha_i-\tfrac12\sum\alpha_i\alpha_jy_iy_jK(x_i,x_j)$ s.t. $0\le\alpha_i\le100,\ \sum\alpha_iy_i=0$. The matrix $H_{ij}=y_iy_jK(x_i,x_j)$:
$$H=\begin{bmatrix}4&9&-25&-36&49\\9&25&-81&-121&169\\-25&-81&289&441&-625\\-36&-121&441&676&-961\\49&169&-625&-961&1369\end{bmatrix}$$
**QP solution:** $\alpha=(0,\ 2.5,\ 0,\ 7.333,\ 4.833)$; check $\sum\alpha_iy_i=2.5-7.333+4.833=0$ ✓. **Support vectors $\{x_2=2,\ x_4=5,\ x_5=6\}$.**
$$f(z)=2.5(1)(2z+1)^2+7.333(-1)(5z+1)^2+4.833(1)(6z+1)^2+b=0.6667z^2-5.333z+b$$
$b$ from $f(2)=1$: $0.6667(4)-5.333(2)+b=1\Rightarrow b=9$ (same from $f(5)=-1$ and $f(6)=1$).
**Answer:** $f(z)=0.6667z^2-5.333z+9$. Check: $f(1)=4.33>0$ (class 1 ✓), $f(4)=-1.67<0$ (class 2 ✓). The sign pattern +,+,−,−,+ is not linearly separable in 1-D, but separable with the quadratic kernel.

---

## N6. Support Vector Regression (linear kernel)

**Given:** points (1,2), (2,3), (3,5); $\varepsilon=0.5$, $C=10$. Candidate line $f(x)=1.5x+0.5$.

| $x$ | $\hat y=1.5x+0.5$ | error $y-\hat y$ | inside tube ($\lvert e\rvert\le0.5$)? | slack |
|---|---|---|---|---|
| 1 | 2.0 | 0.0 | yes | 0 |
| 2 | 3.5 | −0.5 | yes (on lower edge) | 0 |
| 3 | 5.0 | 0.0 | yes | 0 |

Tube: upper $1.5x+1.0$, lower $1.5x+0.0$. All slacks 0 ⇒ penalty term is 0.
$$\text{Cost}=\tfrac12w^2+C\sum(\xi_i+\xi_i^*)=\tfrac12(1.5)^2+10(0)=\mathbf{1.125}$$
(⚠ see the theory-note check about which points are truly support vectors: only (2,3) sits on the tube edge.)

---

## N7. K-Means (4 medicines, K = 2)

**Data:** A(1,1), B(2,1), C(4,3), D(5,4) (weight index, pH).

### N7a. Euclidean, seeds $c_1=A,\ c_2=B$ (slides 6-11)
| Round | Centroids | Distances to $c_1$: A, B, C, D | to $c_2$: A, B, C, D | Membership |
|---|---|---|---|---|
| 1 | $c_1=(1,1),\ c_2=(2,1)$ | 0, 1, 3.61, 5 | 1, 0, 2.83, 4.24 | {A} , {B,C,D} |
| 2 | $c_1=(1,1),\ c_2=(\tfrac{11}3,\tfrac83)$ | 0, 1, 3.61, 5 | 3.14, 2.36, 0.47, 1.89 | {A,B} , {C,D} |
| 3 | $c_1=(1.5,1),\ c_2=(4.5,3.5)$ | 0.5, 0.5, 3.20, 4.61 | 4.30, 3.54, 0.71, 0.71 | {A,B} , {C,D} (**no change → stop**) |

New centroids: $c_2=\big(\tfrac{2+4+5}3,\tfrac{1+3+4}3\big)=(\tfrac{11}3,\tfrac83)$; then $c_1=\big(\tfrac{1+2}2,\tfrac{1+1}2\big)=(1.5,1)$, $c_2=\big(\tfrac{4+5}2,\tfrac{3+4}2\big)=(4.5,3.5)$.
Example distance: $d(D,c_1)=\sqrt{(5-1)^2+(4-1)^2}=5$, $d(D,c_2)=\sqrt{(5-2)^2+(4-1)^2}=4.24$.

### N7b. Exercise: **Manhattan** distance, seeds $c_1=A,\ c_2=C$ (slide 12)
Manhattan $d=\lvert\Delta x\rvert+\lvert\Delta y\rvert$.
* **Round 1** ($c_1=(1,1),c_2=(4,3)$): to $c_1$: A 0, B 1, C 5, D 7; to $c_2$: A 5, B 4, C 0, D 2 ⇒ {A,B}, {C,D}. New centroids $(1.5,1)$ and $(4.5,3.5)$.
* **Round 2:** to $c_1$: A 0.5, B 0.5, C 4.5, D 6.5; to $c_2$: A 6, B 5, C 1, D 1 ⇒ {A,B}, {C,D}: **unchanged → converged.**

**Answers:** (1) **2 iterations** (the 2nd confirms no change); (2) clusters **{A,B}** and **{C,D}**; (3) centroids **(1.5, 1)** and **(4.5, 3.5)**.

---

## N8. Hierarchical Clustering: Italian cities (distance matrix)

| | BA | FI | MI | NA | RM | TO |
|---|---|---|---|---|---|---|
| **BA** | 0 | 662 | 877 | 255 | 412 | 996 |
| **FI** | 662 | 0 | 295 | 468 | 268 | 400 |
| **MI** | 877 | 295 | 0 | 754 | 564 | 138 |
| **NA** | 255 | 468 | 754 | 0 | 219 | 869 |
| **RM** | 412 | 268 | 564 | 219 | 0 | 669 |
| **TO** | 996 | 400 | 138 | 869 | 669 | 0 |

### N8a. Single link (min distance), as in the slides
| Merge | Smallest entry | Merge height | New cluster-to-cluster distances (min rule) |
|---|---|---|---|
| 1 | MI–TO | **138** | MI/TO–BA = min(877,996)=877; –FI = min(295,400)=295; –NA = min(754,869)=754; –RM = min(564,669)=564 |
| 2 | NA–RM | **219** | NA/RM–BA = min(255,412)=255; –FI = min(468,268)=268; –MI/TO = min(754,564)=564 |
| 3 | BA–NA/RM | **255** | BA/NA/RM–FI = min(662,268)=268; –MI/TO = min(877,564)=564 |
| 4 | FI–BA/NA/RM | **268** | BA/FI/NA/RM–MI/TO = min(564,295)=295 |
| 5 | BA/FI/NA/RM – MI/TO | **295** | one cluster left |

Dendrogram (single link): MI+TO (138), NA+RM (219), + BA (255), + FI (268), join with MI/TO (295).

### N8b. Complete link (max rule), extra practice
Merges at **138** (MI–TO), **219** (NA–RM), **400** (FI joins MI/TO: max(295,400)), **412** (BA joins NA/RM: max(255,412)), **996** (final). Complete link gives the later, tighter merges (400/412/996 vs. 268/295 for single link).

---

## N9. Association Rules

### N9a. Support / confidence on a toy set (slide 5)
With min support 50 % and min confidence 50 %: $A\Rightarrow C$ has (support 50 %, confidence 66.6 %); $C\Rightarrow A$ has (50 %, 100 %).

### N9b. Apriori on 9 transactions (slide 11)
| TID | Items | TID | Items |
|---|---|---|---|
| 100 | I1,I2,I5 | 600 | I2,I3 |
| 200 | I2,I4 | 700 | I1,I3 |
| 300 | I2,I3 | 800 | I1,I2,I3,I5 |
| 400 | I1,I2,I4 | 900 | I1,I2,I3 |
| 500 | I1,I3 | | |

*(The slide does not state the threshold; the standard textbook example uses **minimum support count = 2**, which is what is used below.)*

| Level | Candidates → counts | Frequent ($\ge2$) |
|---|---|---|
| $C_1$ → $L_1$ | I1:6, I2:7, I3:6, I4:2, I5:2 | all five |
| $C_2$ → $L_2$ | {I1,I2}:4, {I1,I3}:4, {I1,I4}:1, {I1,I5}:2, {I2,I3}:4, {I2,I4}:2, {I2,I5}:2, {I3,I4}:0, {I3,I5}:1, {I4,I5}:0 | {I1,I2}, {I1,I3}, {I1,I5}, {I2,I3}, {I2,I4}, {I2,I5} |
| $C_3$ (join, then **prune**) | join gives {I1,I2,I3}, {I1,I2,I5}, {I1,I3,I5}, {I2,I3,I4}, {I2,I3,I5}, {I2,I4,I5}. Pruned: {I1,I3,I5} (has {I3,I5}∉$L_2$), {I2,I3,I4} ({I3,I4}), {I2,I3,I5} ({I3,I5}), {I2,I4,I5} ({I4,I5}). Left: {I1,I2,I3}:2, {I1,I2,I5}:2 | both |
| $C_4$ | {I1,I2,I3,I5} pruned ({I2,I3,I5}∉$L_3$) → $L_4=\varnothing$ | **stop** |

*Extra rule example:* $\{I1,I2\}\Rightarrow I5$: confidence $=\text{supp}\{I1,I2,I5\}/\text{supp}\{I1,I2\}=2/4=50\%$.

### N9c. FP-Growth (slide 16-25), min support 0.5 (count ≥ 3 of 5)
| TID | Items bought | Ordered frequent items |
|---|---|---|
| 100 | f,a,c,d,g,i,m,p | f,c,a,m,p |
| 200 | a,b,c,f,l,m,o | f,c,a,b,m |
| 300 | b,f,h,j,o | f,b |
| 400 | b,c,k,s,p | c,b,p |
| 500 | a,f,c,e,l,p,m,n | f,c,a,m,p |

**Header table (frequency):** f 4, c 4, a 3, b 3, m 3, p 3.
**Conditional pattern bases:** c: {f:3}; a: {fc:3}; b: {fca:1, f:1, c:1}; m: {fca:2, fcab:1}; p: {fcam:2, cb:1}.
**Conditional FP-trees:** m → {f:3, c:3, a:3} (single path); p → {c:3}; a → {f:3,c:3}; c → {f:3}; b → empty; f → empty.
**All frequent patterns with m** (single-path combinations): m, fm, cm, am, fcm, fam, cam, **fcam** (all support 3). With p: p, cp.

---

## N10. Evaluation-Metric Numericals

### N10a. Cat vs Dog (20 animals), cat = positive
$TP=6,\ TN=11,\ FP=2\ (\text{Type I}),\ FN=1\ (\text{Type II})$.
$$\text{Acc}=\frac{6+11}{20}=0.85,\quad \text{Prec}=\frac{6}{6+2}=0.75,\quad \text{Recall}=\frac{6}{6+1}=0.857,\quad F_1=\frac{2PR}{P+R}=0.80,\quad \text{Spec}=\frac{11}{11+2}=0.846$$
> ⚠ **Check:** the slide shows $F_1=2\cdot\frac{0.85\times0.75}{0.85+0.75}=0.79$ (recall rounded to 0.85). Exact $F_1=0.80$.

### N10b. Multi-class (urgent / normal / spam), rows = system output, columns = gold
$$\begin{array}{c|ccc}&\text{urgent}&\text{normal}&\text{spam}\\\hline\text{urgent}&8&10&1\\\text{normal}&5&60&50\\\text{spam}&3&30&200\end{array}$$
Precision: urgent $8/19=.42$, normal $60/115=.52$, spam $200/233=.86$. Recall: urgent $8/16=.50$, normal $60/100=.60$, spam $200/251=.80$.
**Macro-average precision** $=(.42+.52+.86)/3=\mathbf{.60}$.
**Micro-average:** pool the per-class 2×2 tables → TP = 8+60+200 = **268**, FP = 11+55+33 = **99**, FN = **99**, TN = **635**; micro precision $=268/(268+99)=\mathbf{.73}$ (dominated by the big "spam" class).

### N10c. Python `classification_report` (Aeroplane / Boat / Car)
| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Aeroplane | 0.67 | 0.67 | 0.67 | 3 |
| Boat | 0.25 | 1.00 | 0.40 | 1 |
| Car | 1.00 | 0.50 | 0.67 | 6 |
| accuracy | | | 0.60 | 10 |
| macro avg | 0.64 | 0.72 | 0.58 | 10 |
| weighted avg | 0.82 | 0.60 | 0.64 | 10 |

Check: macro precision $=(0.67+0.25+1.0)/3=0.64$; weighted precision $=(0.67\cdot3+0.25\cdot1+1.0\cdot6)/10=0.82$; weighted recall $=(0.67\cdot3+1.0\cdot1+0.5\cdot6)/10=0.60$.

### N10d. Regression errors
Actual $[2,3,9,6]$, predicted $[1,2,8,1]$: errors $[1,1,1,5]$.
$\text{MAE}=(1+1+1+5)/4=\mathbf2$; squared errors $[1,1,1,25]$ → $\text{MSE}=28/4=\mathbf7$; $\text{RMSE}=\sqrt7=\mathbf{2.65}$. (The single error of 5 contributes 25 of the 28, so MSE/RMSE are far more outlier-sensitive than MAE.)
$R^2=0.85$ ⇒ 85 % of target variance explained by the model.

### N10e. Precision/recall vs threshold (Imbalance slides 17-28)
*Basic:* labels vs predictions → precision $=2/4$, recall $=2/3$.
*Threshold sweep* (7 test examples, 3 true positives, sorted by confidence; classify the top-$k$ as positive):

| top-k | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| precision | 1.0 | 0.5 | 0.66 | 0.5 | 0.6 | 0.5 | 0.43 |
| recall | 0.33 | 0.33 | 0.66 | 0.66 | 1.0 | 1.0 | 1.0 |

Lowering the threshold (larger $k$) raises recall (to 1.0) while precision falls (to $3/7$).

---

## N11. Imbalanced Data Numbers

* **Phishing:** 30 positives in 1 M emails → always predicting "not phishing" gives **99.997 % accuracy** with recall 0.
* **Weighted examples:** negative weight 1, positive weight $=99.997/0.003\approx\mathbf{33\,332}$.
* **Weighted log-loss:** `class_weight={0:1,1:20}` ⇒ minority error costs 20× ($\text{NewLoss}=-20\,y\log p-(1-y)\log(1-p)$).
* **Abalone-19** (4174 rows; 99.23 % negative / 0.77 % positive; stratified 80/20 split: train 3313 neg + 26 pos, test 829 neg + 6 pos). Logistic Regression **AUC on test:**

| Training data | AUC |
|---|---|
| Original (imbalanced) | 0.684 |
| Random over-sampling (3313 / 3313) | **0.839** |
| SMOTE (3313 / 3313) | 0.791 |
| Random under-sampling (26 / 26) | 0.647 (worse: too little data) |

---

## N12. Self-Training (movie reviews)
**Labelled:** "Loved the plot, amazing acting!" (+); "Boring, waste of time." (−). **Unlabelled:** A "Amazing visuals, loved it."; B "Terrible script, very boring."; C "The acting was okay, plot was average."

| Review | Prediction | Confidence | Threshold ≥ 85 % |
|---|---|---|---|
| A | + | 95 % | **pass** → pseudo-label + |
| B | − | 90 % | **pass** → pseudo-label − |
| C | + | 55 % | **fail** → stays unlabelled |

Move A and B into $L$ (4 labelled examples), **retrain**; with new words ("visuals", "terrible/script") the confidence for C will likely exceed 85 % next round. Threshold 50 % would risk reinforcing mistakes.

---

## N13. Label Propagation (4 nodes, 2 classes)

**Given:** node 1 labelled Class 0, node 2 labelled Class 1, nodes 3, 4 unlabelled.
$$W=\begin{bmatrix}0&2&1&0\\2&0&0&4\\1&0&0&3\\0&4&3&0\end{bmatrix},\quad \text{row sums}=(3,6,4,7),\quad P=D^{-1}W=\begin{bmatrix}0&\tfrac23&\tfrac13&0\\\tfrac26&0&0&\tfrac46\\\tfrac14&0&0&\tfrac34\\0&\tfrac47&\tfrac37&0\end{bmatrix}$$
$$Y^{(0)}=\begin{bmatrix}1&0\\0&1\\0&0\\0&0\end{bmatrix}$$
**Iteration 1:** $Y=PY^{(0)}$; rows 3,4 → $[0.25,\,0]$ and $[0,\,0.57]$; **clamp** rows 1,2 back to $[1,0]$, $[0,1]$.
**Iteration 2:** node 3 $=0.25\,[1,0]+0.75\,[0,0.57]=[0.25,\,0.43]$ (row 3 of $P$ is $(0.25,0,0,0.75)$); node 4 $=0.57\,[0,1]+0.43\,[0.25,0]=[0.11,\,0.57]$ (row 4 of $P$ is $(0,0.57,0.43,0)$, using the node-3 value $[0.25,0]$ from iteration 1). Clamp rows 1,2 again.
**Converged:** node 3 $=[0.368,\,0.632]$, node 4 $=[0.158,\,0.842]$ (solve $a_3=0.25+0.75a_4,\ a_4=0.4286a_3$).
**Prediction:** node 3 → **Class 1** (0.632 > 0.368), node 4 → **Class 1** (0.842 > 0.158).

> ⚠ **Check:** the professor's "final" matrix shows [0.32, 0.68] and [0.14, 0.86] (an approximate, row-normalised stop). The true limit is [0.37, 0.63] and [0.16, 0.84]. **Predictions are identical (Class 1 for both).**

---

## N14. Label Spreading (3 nodes, 2 classes, $\alpha=0.5$)

**Given:** node 0 = Class 0, node 1 unlabelled, node 2 = Class 1. Edges: 0–1 weight 2.0, 1–2 weight 1.0, 0–2 none.
$$Y=\begin{bmatrix}1&0\\0&0\\0&1\end{bmatrix},\ W=\begin{bmatrix}0&2&0\\2&0&1\\0&1&0\end{bmatrix},\ D=\operatorname{diag}(2,3,1),\ D^{-1/2}=\operatorname{diag}(0.707,0.577,1)$$
$$S=D^{-1/2}WD^{-1/2}=\begin{bmatrix}0&0.816&0\\0.816&0&0.577\\0&0.577&0\end{bmatrix}\quad(S_{01}=\tfrac{2}{\sqrt{2\cdot3}},\ S_{12}=\tfrac1{\sqrt{3\cdot1}})$$
**Iteration 1:** $F^{(1)}=0.5\,SY+0.5\,Y$. $SY=\begin{bmatrix}0&0\\0.816&0.577\\0&0\end{bmatrix}$ (node 1 picks up node 0's class-0 and node 2's class-1; nodes 0 and 2 receive nothing because node 1 has no label yet).
$$F^{(1)}=\begin{bmatrix}0.5&0\\0.408&0.289\\0&0.5\end{bmatrix}\ \Rightarrow\ \text{node 1: }\arg\max(0.408,0.289)=\text{Class 0}$$
**Iteration 2:** $SF^{(1)}=\begin{bmatrix}0.333&0.236\\0.408&0.289\\0.236&0.167\end{bmatrix}$, so
$$F^{(2)}=\begin{bmatrix}0.667&0.118\\0.204&0.144\\0.118&0.583\end{bmatrix}\ \Rightarrow\ \text{node 1: }\arg\max(0.204,0.144)=\text{Class 0}$$
**Converged** (stop when change $<10^{-3}$): node 1 ≈ $[0.272,\,0.193]$ ⇒ **Class 0** (it sits closer, with the heavier edge weight 2.0, to the Class-0 node).

> ⚠ **Check:** the professor's PDF computes $S\cdot F^{(0)}$ with a third row $[0,\,0.577]$ (should be $[0,0]$), giving $F^{(1)}=[\dots;0,\,0.789]$ and then "Class 1 at iteration 2". With the correct arithmetic, node 1 is **Class 0 at both iterations and at convergence**. The method (formula, matrices $D$, $D^{-1/2}$, $S$) is right; only the multiplication slipped.

---

## N15. GNN: one message-passing step
**Given:** node A with neighbours B, C; $N(A)=\{A,B,C\}$; $h_A=[1,0],\ h_B=[0,2],\ h_C=[3,1]$; $W=\begin{bmatrix}2&1\\0&2\end{bmatrix}$.
1. **Mean aggregation:** $h_{N(A)}=\tfrac13([1,0]+[0,2]+[3,1])=[1.33,\ 1.00]$.
2. **Transform:** $z=W\,h_{N(A)}=[2(1.33)+1(1.00),\ 0(1.33)+2(1.00)]=[3.67,\ 2.00]$.
3. **ReLU:** $\max(0,z)=[3.67,\ 2.00]$ (unchanged, both positive).

> ⚠ **Check:** the document prints 3.66 (using 1.33 rounded). Exact $= 11/3=3.67$.

---

## N16. PCA

### N16a. 5-point example (slides; population covariance, divide by $N$)
**Points:** P1(2,2), P2(3,1), P3(4,3), P4(5,5), P5(6,4). Means $(4,\,3)$.
Centred: $(-2,-1),(-1,-2),(0,0),(1,2),(2,1)$.
$\text{Var}(X)=10/5=2,\ \text{Var}(Y)=10/5=2,\ \text{Cov}=8/5=1.6$ ⇒ $S=\begin{bmatrix}2&1.6\\1.6&2\end{bmatrix}$.
**Characteristic equation:** $(2-\lambda)^2-1.6^2=0\Rightarrow\lambda_1=3.6,\ \lambda_2=0.4$.
**Eigenvectors:** $\lambda_1$: $-1.6x+1.6y=0\Rightarrow x=y$, $u_1=[0.7071,\,0.7071]$. $\lambda_2$: $x=-y$, $u_2=[-0.7071,\,0.7071]$.
**Variance explained by PC1** $=3.6/(3.6+0.4)=\mathbf{90\%}$.
**Scores on PC1** $=X_c\cdot u_1$: P1 $=-2.1213$, P2 $=-2.1213$, P3 $=0$, P4 $=2.1213$, P5 $=2.1213$.
**Reconstruction** $=\text{score}\cdot u_1^T+\text{mean}$: P1,P2 → $(2.5,\,1.5)$; P3 → $(4,\,3)$; P4,P5 → $(5.5,\,4.5)$. P1/P2 (and P4/P5) collapse to the same point: that is the information lost by dropping PC2 (10 % of variance).

### N16b. Handwritten solved example (sample covariance, divide by $N-1$)
**Data** ($n=2$ features, $N=4$ samples): feature $a$: 4, 8, 13, 7; feature $b$: 11, 4, 5, 14.
**Means:** $\bar a=\frac{4+8+13+7}4=8$, $\bar b=\frac{11+4+5+14}4=8.5$.
**Covariances** (divisor $N-1=3$):
$\text{cov}(a,a)=\frac{16+0+25+1}3=14$; $\text{cov}(a,b)=\frac{(-4)(2.5)+0+(5)(-3.5)+(-1)(5.5)}3=\frac{-33}3=-11$; $\text{cov}(b,b)=\frac{6.25+20.25+12.25+30.25}3=23$.
$$S=\begin{bmatrix}14&-11\\-11&23\end{bmatrix}$$
**Eigenvalues:** $(14-\lambda)(23-\lambda)-121=0\Rightarrow\lambda^2-37\lambda+201=0\Rightarrow\lambda_1=30.38,\ \lambda_2=6.62$.
**Eigenvector for $\lambda_1$** (from $(14-\lambda_1)u_1-11u_2=0$, set ratio $u_1/11=u_2/(14-\lambda_1)=1$): $v=(11,\ 14-30.38)=(11,\,-16.38)$, length $\sqrt{11^2+16.38^2}=19.73$ ⇒ **$n_1=(0.5575,\,-0.8302)$**.
**Eigenvector for $\lambda_2$:** $v=(11,\ 14-6.62)=(11,\ 7.38)$, length 13.24 ⇒ **$n_2=(0.8308,\,0.5574)$**. ($n_1\perp n_2$.)
**PC1 scores** $=n_1^T(x-\bar x)$:
* Sample 1: $(0.5575)(-4)+(-0.8302)(2.5)=-2.230-2.076=\mathbf{-4.305}$
* Sample 2: $(0.5575)(0)+(-0.8302)(-4.5)=\mathbf{3.736}$
* Sample 3: $(0.5575)(5)+(-0.8302)(-3.5)=2.788+2.906=\mathbf{5.692}$
* Sample 4: $(0.5575)(-1)+(-0.8302)(5.5)=-0.558-4.566=\mathbf{-5.123}$

**Explained variance:** PC1 $=30.38/37=82.1\%$, PC2 $=17.9\%$.
**Scree-plot note (handwritten):** with six PCs A–F explaining 0.60, 0.25, 0.15, ... : $0.60+0.25+0.15=0.90$ ⇒ **3 PCs suffice** (≥ 80 % rule of thumb). *PCA scores are what you use for classification/regression.*

---

## N17. LLE Worked Example

### Local weights ($X_i=[2,3,5]^T$, 3 neighbours)
Neighbours $X_1=[2,4,5]$, $X_2=[1,3,5]$, $X_3=[2,3,7]$.
Displacements $X_i-X_j$: $[0,-1,0],\ [1,0,0],\ [0,0,-2]$.
Local covariance $G_{jk}=(X_i-X_j)^T(X_i-X_k)$ ⇒ $G=\begin{bmatrix}1&0&0\\0&1&0\\0&0&4\end{bmatrix}$.
Solve $Gw'=\mathbf 1$: $w'=[1,\ 1,\ 0.25]$; normalise by the sum $2.25$:
$$W_1=0.444,\quad W_2=0.444,\quad W_3=0.111\quad(\text{sum}=1\ \checkmark)$$

### Global cost matrix $M=(I-W)^T(I-W)$ (4 points)
$$W=\begin{bmatrix}0&0.6&0.4&0\\0.5&0&0&0.5\\0.3&0&0&0.7\\0&0.8&0.2&0\end{bmatrix}\ \Rightarrow\ M=\begin{bmatrix}1.34&-1.10&-0.70&0.46\\-1.10&2.00&0.40&-1.30\\-0.70&0.40&1.20&-0.90\\0.46&-1.30&-0.90&1.74\end{bmatrix}$$
Example cell: $M_{11}=1^2+(-0.5)^2+(-0.3)^2+0^2=1.34$; $M_{12}=(1)(-0.6)+(-0.5)(1)+(-0.3)(0)+(0)(-0.8)=-1.10$.
**Properties:** symmetric; every row/column sums to 0 (e.g. $1.34-1.10-0.70+0.46=0$) ⇒ smallest eigenvalue is 0 with eigenvector $[1,1,1,1]^T$ ⇒ **discard it**; for a 2-D embedding take the **2nd and 3rd smallest eigenvectors** (their entries are the coordinates $Y_i$).

---

## N18. Q-Learning and Reinforcement Learning Numericals (N18a-N18j)

### N18a. Single-step Q-learning update
$Q(S_1,A_1)=12$, reward $R=5$, next state $S_2$ has $Q(S_2,\text{Up})=20,\ Q(S_2,\text{Down})=30$; $\alpha=0.2,\ \gamma=0.8$.
$\max_{a'}Q(S_2,a')=30$; TD target $=5+0.8(30)=29$.
$$Q(S_1,A_1)\leftarrow(1-\alpha)Q+\alpha[\text{target}]=0.8(12)+0.2(29)=9.6+5.8=\mathbf{15.4}$$
(Equivalent: $12+0.2(29-12)=15.4$.)

### N18b. 3-state chain A → B → C (C terminal), $\alpha=0.5,\ \gamma=0.9$, all $Q=0$, reward +100 for entering C, else 0
| Step | Move | Update | New value |
|---|---|---|---|
| 1 | A –R→ B, $r=0$ | $Q(A,R)=0.5(0)+0.5[0+0.9\cdot0]$ | $Q(A,R)=0$ |
| 2 | B –R→ C, $r=100$ | $Q(B,R)=0.5(0)+0.5[100+0.9\cdot0]$ | $Q(B,R)=\mathbf{50}$ |
| 3 | A –R→ B, $r=0$ | $Q(A,R)=0.5(0)+0.5[0+0.9\cdot50]$ | $Q(A,R)=\mathbf{22.5}$ |

Final table: $Q(A,R)=22.5$, $Q(B,R)=50.0$, all other pairs 0.

### N18c. 4-state chain $s_1\to s_2\to s_3\to s_4$ (goal), $\gamma=0.9,\ \alpha=1$ (so $Q=R+\gamma\max Q'$), reward 100 for entering $s_4$
* Episode 1 from $s_3$: $Q(s_3,R)=100+0.9(0)=\mathbf{100}$.
* Episode 2 from $s_2$: $Q(s_2,R)=0+0.9\max(0,100)=\mathbf{90}$; then $s_3$: 100 again.
* Episode 3 from $s_1$: $Q(s_1,R)=0+0.9\max(0,90)=\mathbf{81}$.
* Non-optimal "Left" values after convergence: $Q(s_2,L)=0+0.9\max Q(s_1,\cdot)=0.9(81)=\mathbf{72.9}$; $Q(s_3,L)=0.9(90)=81$; $Q(s_1,L)=0$ (no state to the left, stays 0).

| State | Left | Right | Optimal action |
|---|---|---|---|
| $s_1$ | 0 | 81 | Right |
| $s_2$ | 72.9 | 90 | Right |
| $s_3$ | 81 | 100 | Right |
| $s_4$ | terminal | terminal | – |

General pattern: $Q=100\gamma^{k}$ where $k$ = steps to the goal minus 1 (100, 90, 81).

### N18d. Model-free vs model-based on a 3-tile hallway $S_1\to S_2\to G$, reward +10 entering G, $\gamma=0.9$
* **Model-free (Q-learning):** Trial 1: $S_1\to S_2$ ($r=0$): $Q(S_1,R)=0+\gamma\cdot0=0$. Then $S_2\to G$ ($r=10$): $Q(S_2,R)=10+0.9(0)=10$. Trial 2: $Q(S_1,R)=0+0.9(10)=\mathbf9$. A state $N$ steps from the goal needs about $N$ real trials for the reward to back-propagate.
* **Model-based (planning):** after **one** trial it stores $(S_1,R)\to S_2$, $(S_2,R)\to G$, $R(S_2,R)=10$, then *simulates*: $V(S_2)=10$, $V(S_1)=0+0.9(10)=\mathbf9$ with no second real trial.

### N18e. Optimal policy by Bellman ($\gamma=0.5$): Leave (+10, terminal) vs Stay (+2, back to $S$)
$Q(S,\text{Leave})=10+0.5(0)=10$. $V^*(S)=\max\{10,\ 2+0.5V^*(S)\}$. If Leave is best, $V^*(S)=10$ and then $2+0.5(10)=7<10$ ✓ consistent. **$\pi^*(S)=\text{Leave}$.**
> ⚠ **Check:** the PDF states $Q(S,\text{Stay})=4$ (from assuming Stay is chosen forever: $V=2+0.5V\Rightarrow4$). Under the optimal value the correct $Q^*(S,\text{Stay})=7$. Either way $10>4$ and $10>7$, so **Leave** is optimal.

### N18f. Value propagation, "Q values never decrease" (Alpaydin Fig 18.4, $\gamma=0.9$, goal reward 100)
Path A seen first: $Q(*)=0.9\max(0,81)=72.9\approx73$. Then path B (shorter): $0.9\max(100,81)=90$. Reverse order gives $0.9\max(100,0)=90$ and seeing A afterwards leaves it at 90.

> ★ **N18g-N18j below are extra Q-learning / value-based practice problems I built** (the professor's files have N18a-N18f). Method is identical; answers recomputed in Python.

### N18g. Full Q-table trace (2×2 grid, $\alpha=0.5,\ \gamma=0.9$)
**Grid:** A (top-left), B (top-right), C (bottom-left), **G** (bottom-right, terminal). Moving **into G gives $+10$**, every other move $0$. Actions: R (right), D (down). All $Q=0$ at the start. Observed moves, in order:

| # | Move | $\max_{a'}Q(s',\cdot)$ | TD target $R+\gamma\max Q'$ | New $Q=Q+\alpha(\text{target}-Q)$ |
|---|---|---|---|---|
| 1 | C –R→ G, $R=10$ | 0 (terminal) | $10$ | $Q(C,R)=0+0.5(10-0)=\mathbf{5}$ |
| 2 | A –D→ C, $R=0$ | $\max(Q(C,R),Q(C,D))=5$ | $0+0.9(5)=4.5$ | $Q(A,D)=0+0.5(4.5)=\mathbf{2.25}$ |
| 3 | C –R→ G, $R=10$ | 0 | $10$ | $Q(C,R)=5+0.5(10-5)=\mathbf{7.5}$ |
| 4 | A –D→ C, $R=0$ | $7.5$ | $0.9(7.5)=6.75$ | $Q(A,D)=2.25+0.5(6.75-2.25)=\mathbf{4.5}$ |
| 5 | B –D→ G, $R=10$ | 0 | $10$ | $Q(B,D)=0+0.5(10)=\mathbf5$ |
| 6 | A –R→ B, $R=0$ | $5$ | $0.9(5)=4.5$ | $Q(A,R)=0+0.5(4.5)=\mathbf{2.25}$ |

**Result:** $Q(A,D)=4.5>Q(A,R)=2.25$ ⇒ greedy policy at A is **Down**; $Q(C,R)=7.5$; $Q(B,D)=5$. Note how the value of A only rose **after** C had a non-zero value (reward propagates backwards), and that $Q(C,R)\to10$ as it is visited more ($5,\,7.5,\,8.75,\,9.375,\dots$).

### N18h. Q-learning vs SARSA on the same transition
**Given:** $Q(s,a)=2$, $R=1$, $\gamma=0.9$, $\alpha=0.5$. In the next state $s'$: $Q(s',\text{left})=4,\ Q(s',\text{right})=8$.
* **Q-learning (off-policy):** target $=1+0.9\max(4,8)=1+7.2=8.2$ ⇒ $Q=2+0.5(8.2-2)=\mathbf{5.1}$ (always uses the best action, "right").
* **SARSA (on-policy):** the ε-greedy policy actually picked $a'=$ left (an exploratory move): target $=1+0.9(4)=4.6$ ⇒ $Q=2+0.5(4.6-2)=\mathbf{3.3}$. If it had picked "right", SARSA would also give 5.1.
**Moral:** Q-learning ignores the exploratory step; SARSA includes its cost, so it is more cautious.

### N18i. Value iteration (value-based RL, known model), $\gamma=0.9$
**MDP:** state $S_1$: action *Stay* → $S_1$ with reward 1; action *Go* → $S_2$ with reward 0. State $S_2$: only *Stay* → $S_2$ with reward 2. Start $V=0$.
Update: $V(S)\leftarrow\max_a[R+\gamma V(S')]$.

| Iter | $Q(S_1,\text{Stay})=1+0.9V(S_1)$ | $Q(S_1,\text{Go})=0.9V(S_2)$ | $V(S_1)$ | $V(S_2)=2+0.9V(S_2)$ | greedy at $S_1$ |
|---|---|---|---|---|---|
| 1 | 1.0 | 0 | **1.0** | **2.0** | Stay |
| 2 | 1.9 | 1.8 | **1.9** | **3.8** | Stay |
| 3 | 2.71 | 3.42 | **3.42** | **5.42** | **Go** |
| 4 | 4.078 | 4.878 | **4.878** | **6.878** | Go |

**Converged:** $V^*(S_2)=2/(1-0.9)=20$; $V^*(S_1)=\max\{1/(1-0.9)\ (\text{stay forever})=10,\ 0.9(20)=18\}=\mathbf{18}$ ⇒ $\pi^*(S_1)=\text{Go}$ (check: $Q^*(S_1,\text{Stay})=1+0.9(18)=17.2<18$). The **policy turned correct at iteration 3, long before the values converged**.

### N18j. TD(0), ε-greedy and softmax probabilities
* **TD(0) state-value update:** $V(s)=5,\ R=2,\ V(s')=10,\ \alpha=0.1,\ \gamma=0.9$ ⇒ TD error $\delta=2+0.9(10)-5=6$ ⇒ $V(s)=5+0.1(6)=\mathbf{5.6}$.
* **ε-greedy**, 4 actions, $\varepsilon=0.1$ (random action drawn from **all 4**): best action probability $=1-\varepsilon+\varepsilon/4=\mathbf{0.925}$; each other action $=\varepsilon/4=0.025$. (If the random draw excludes the best action, the best has simply $1-\varepsilon=0.9$.)
* **Softmax**, $Q=[2,1,0]$: $P_i=e^{Q_i/T}/\sum_je^{Q_j/T}$.

| Temperature $T$ | $P(a_1)$ | $P(a_2)$ | $P(a_3)$ | behaviour |
|---|---|---|---|---|
| 10 | 0.367 | 0.332 | 0.301 | near-uniform (**explore**) |
| 1 | 0.665 | 0.245 | 0.090 | balanced |
| 0.5 | 0.867 | 0.117 | 0.016 | mostly best |
| 0.1 | 1.000 | 0.000 | 0.000 | greedy (**exploit**) |

<div class="pb"></div>

# PART B: Numericals for Syllabus Topics Not Covered by the Professor's Files

> ⚠ **Source note:** N19-N29 are **practice problems built by me** for topics with no file in Prof Provided Notes (random forest/boosting, polynomial & regularised regression, bias-variance, policy gradients, Dyna-Q, PCA variance, extra metrics, cross-validation, grid search). Every answer was recomputed in Python/scikit-learn. They show the **method** examiners use; the numbers are mine.

---

## N19. Bootstrap, Random Forest and AdaBoost (Module 1)

**(a) Bootstrap / OOB.** Probability a given row is *not* in a bootstrap sample of size $n$: $(1-\frac1n)^n$.
$n=10$: 0.349; $n=100$: 0.366; $n=1000$: 0.3677; $n\to\infty$: $e^{-1}=\mathbf{0.368}$. So ≈ **63.2 %** of distinct rows are in-bag, **36.8 %** are out-of-bag (used for the free OOB error estimate).

**(b) Variance of a bagged ensemble.** Single tree variance $\sigma^2=4$, pairwise correlation $\rho=0.5$, $B=10$ trees:
$$\text{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2=0.5(4)+\frac{0.5}{10}(4)=2+0.2=\mathbf{2.2}$$
Variance fell from 4 to 2.2; with $\rho=0$ it would be $4/10=0.4$ (why random forests decorrelate trees by random feature choice).

**(c) One AdaBoost round.** 10 training points, initial weights $w_i=0.1$; the weak learner misclassifies 2 points.
1. Weighted error $\varepsilon=0.1+0.1=\mathbf{0.2}$.
2. Learner weight $\alpha=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}=\tfrac12\ln4=\mathbf{0.693}$.
3. Update $w_i\leftarrow w_i\,e^{\mp\alpha}$ (× $e^{+\alpha}=2$ if wrong, × $e^{-\alpha}=0.5$ if right): wrong → $0.2$ each; right → $0.05$ each. Sum $Z=2(0.2)+8(0.05)=0.8$.
4. Normalise ($\div0.8$): wrong → **0.25** each, right → **0.0625** each. The 2 hard points now carry **50 %** of the total weight, forcing the next learner to focus on them.
Final prediction: $H(x)=\operatorname{sign}\big(\sum_t\alpha_th_t(x)\big)$.

**(d) One gradient-boosting step** (squared loss, $\nu=0.5$). Targets $y=[3,5,9]$. $F_0=\bar y=5.667$; residuals $r=y-F_0=[-2.667,\,-0.667,\,3.333]$. If the tree fits the residuals exactly: $F_1=F_0+\nu h=[4.333,\ 5.333,\ 7.333]$ (moved halfway toward $y$).

---

## N20. Polynomial Regression: choosing the degree

**Given:** $x=0,1,2,3,4$; $y=1,3,7,13,21$.
* **Degree 1:** $\hat y=5x-1$. Residuals $y-\hat y=[2,-1,-2,-1,2]$ ⇒ $SSE=4+1+4+1+4=\mathbf{14}$ (underfit).
* **Degree 2:** least squares gives $\hat y=x^2+x+1$ exactly ⇒ residuals 0, $SSE=\mathbf0$ (check: $x=3\Rightarrow9+3+1=13$ ✓).
* **Degree 4 with 5 points** would also hit $SSE=0$ but would just memorise noise: choose the **lowest degree that fits well**, validated on held-out data.

---

## N21. Ridge vs Lasso (one feature, no intercept)

**Given:** $x=[1,2,3,4]$, $y=[2,4,5,8]$; $\sum xy=57$, $\sum x^2=30$. **OLS:** $\hat\beta=57/30=\mathbf{1.90}$.

**Ridge** (minimise $\sum(y-\beta x)^2+\lambda\beta^2$): $\hat\beta=\dfrac{\sum xy}{\sum x^2+\lambda}$
* $\lambda=10$: $57/40=\mathbf{1.425}$; $\lambda=30$: $57/60=\mathbf{0.95}$; as $\lambda\to\infty$: $\to0$ but **never exactly 0**.

**Lasso** (minimise $\sum(y-\beta x)^2+\lambda\lvert\beta\rvert$): soft-threshold $\hat\beta=\dfrac{\operatorname{sign}(\sum xy)\,\max(\lvert\sum xy\rvert-\lambda/2,\ 0)}{\sum x^2}$
* $\lambda=10$: $(57-5)/30=\mathbf{1.733}$; $\lambda=30$: $(57-15)/30=\mathbf{1.40}$; $\lambda=114$ (≥ $2\lvert\sum xy\rvert$): $\hat\beta=\mathbf0$ **exactly**.

**Take-away:** ridge **shrinks smoothly**, lasso **zeroes out** coefficients (feature selection). *(scikit-learn's `Lasso(alpha)` scales the RSS by $1/2n$, so its `alpha` $=\lambda/2n$; checked: `Ridge(alpha=10)` gives 1.425 and `Lasso(alpha=1.25)` gives 1.733.)*

---

## N22. Bias-Variance from repeated predictions

True value $f=10$. Model A predicts $[9,11,12,8,10]$ over 5 training sets; Model B predicts $[7.8,8.2,8.0,7.9,8.1]$.

| | mean | Bias² $=(\bar p-10)^2$ | Variance | Bias² + Var | MSE (check) |
|---|---|---|---|---|---|
| **A** | 10.0 | 0 | 2.00 | **2.00** | 2.00 |
| **B** | 8.0 | 4 | 0.02 | **4.02** | 4.02 |

A is unbiased but unstable (**high variance**, e.g. a deep tree); B is stable but systematically off (**high bias**, e.g. too-simple model). A has the lower total error here. Total expected error $=\text{Bias}^2+\text{Variance}+\sigma^2_\varepsilon$.

---

## N23. Policy Gradient: one REINFORCE update (softmax, 2 actions)

**Given:** $\theta=[0,0]$ ⇒ $\pi=\text{softmax}(\theta)=[0.5,0.5]$. The agent samples action $a=0$ and the episode return is $G=10$; learning rate $\alpha=0.1$.
1. $\nabla_\theta\log\pi(a)=\mathbf 1_a-\pi=[1,0]-[0.5,0.5]=[0.5,\,-0.5]$.
2. $\theta\leftarrow\theta+\alpha\,G\,\nabla\log\pi=[0,0]+0.1(10)[0.5,-0.5]=\mathbf{[0.5,\,-0.5]}$.
3. New policy $\pi=\text{softmax}([0.5,-0.5])=\mathbf{[0.731,\ 0.269]}$ (probability of the rewarded action rose 0.5 → 0.73).
With a smaller return $G=4$: $\theta=[0.2,-0.2]$, $\pi=[0.599,0.401]$ (smaller push). **A negative $G$ (or advantage) would lower** the chosen action's probability.

---

## N24. PPO Clipping, UCB and DQN Target

**(a) PPO clipped objective**, $\epsilon=0.2$ ($\text{clip}$ range $[0.8,1.2]$), contribution $\min(rA,\ \text{clip}(r)A)$:

| ratio $r$ | advantage $A$ | $rA$ | $\text{clip}(r)A$ | **min (used)** | comment |
|---|---|---|---|---|---|
| 1.5 | +2 | 3.0 | 2.4 | **2.4** | gain capped: don't push probability up too far |
| 1.5 | −2 | −3.0 | −2.4 | **−3.0** | pessimistic bound kept: bad action must still be penalised |
| 0.5 | +2 | 1.0 | 1.6 | **1.0** | |
| 0.5 | −2 | −1.0 | −1.6 | **−1.6** | |
| 1.1 | +2 | 2.2 | 2.2 | **2.2** | inside the trust region: unclipped |

**(b) MCTS selection (UCB1):** node value $\bar Q=0.6$, parent visits $N=100$, node visits $n=10$, $c=1.41$:
$\text{UCB}=0.6+1.41\sqrt{\ln100/10}=0.6+1.41(0.679)=\mathbf{1.557}$ (exploration bonus ≈ 0.96 shrinks as $n$ grows).

**(c) DQN target:** $r=1$, $\gamma=0.9$, $\max_{a'}Q_{target}(s',a')=5$ ⇒ $y=1+0.9(5)=\mathbf{5.5}$; loss $=(y-Q(s,a))^2$.

---

## N25. Dyna-Q: planning steps amplify one real experience

**Given:** the agent really experienced $(s,a)\to r=100$, next state terminal. Currently $Q(s,a)=50$; $\alpha=0.5$, $\gamma=0.9$. The model stores this transition, so each **planning** step replays it:
$$Q\leftarrow Q+\alpha[\,r+\gamma\cdot0-Q\,]$$
| Update | $Q(s,a)$ |
|---|---|
| start | 50 |
| after planning step 1 | $50+0.5(100-50)=\mathbf{75}$ |
| step 2 | $75+0.5(25)=\mathbf{87.5}$ |
| step 3 | $87.5+0.5(12.5)=\mathbf{93.75}$ |

With $n=3$ planning steps a single real transition moved $Q$ from 50 toward 100 as much as **three** real visits would (the true value is 100). Plain Q-learning (no planning) would have done only the first update, 75.

---

## N26. PCA: how many components? (and SVD check)

**Eigenvalues** of a 5-feature covariance matrix: $[4,\,2,\,1,\,0.5,\,0.5]$ (sum 8).
Explained-variance ratio $\lambda_i/\sum\lambda$: $[0.50,\ 0.25,\ 0.125,\ 0.0625,\ 0.0625]$; **cumulative** $=[0.50,\ 0.75,\ \mathbf{0.875},\ 0.9375,\ 1.0]$.
* For ≥ 80 % keep **3 PCs** (87.5 %). For ≥ 90 % keep **4 PCs** (93.75 %). Kaiser rule (keep $\lambda>$ mean $=1.6$): keep **2** (4 and 2).

**SVD check on the professor's hand example** (4 samples, features $a,b$; centred matrix $X_c$ of size $4\times2$): singular values $\sigma=[9.5475,\ 4.4548]$ ⇒ $\lambda_i=\sigma_i^2/(N-1)=[30.385,\ 6.615]$, identical to the eigenvalues of $S=\begin{bmatrix}14&-11\\-11&23\end{bmatrix}$ ✓.

---

## N27. Extra classification / regression metrics

Cat-vs-dog matrix from N10a: $TP=6,\ TN=11,\ FP=2,\ FN=1$.
* **Balanced accuracy** $=\frac{TPR+TNR}{2}=\frac{0.857+0.846}{2}=\mathbf{0.852}$.
* **MCC** $=\dfrac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}=\dfrac{66-2}{\sqrt{8\cdot7\cdot13\cdot12}}=\dfrac{64}{93.46}=\mathbf{0.685}$.
* **Adjusted $R^2$:** $R^2=0.85$, $n=50$ samples, $p=3$ predictors: $1-(1-R^2)\dfrac{n-1}{n-p-1}=1-0.15\cdot\dfrac{49}{46}=\mathbf{0.840}$.
* **MAPE** for actual $[2,3,9,6]$, predicted $[1,2,8,1]$: $\frac14\big(\frac12+\frac13+\frac19+\frac56\big)\times100=\mathbf{44.4\%}$ (dominated by the 5-unit miss on a small true value of 6).

---

## N28. Cross-validation arithmetic

**(a) Fold sizes.** $n=10$ samples. 5-fold: 5 test sets of 2 (every sample tested once; 5 fits). 3-fold: test sizes $4,3,3$ (`KFold` puts the remainder in the first folds). **LOO:** 10 fits. **Leave-2-Out:** $\binom{10}{2}=\mathbf{45}$ fits.

**(b) Stratified 5-fold.** 100 samples, 90 negative / 10 positive. Each fold has 20 samples with **18 negative + 2 positive** (ratio preserved). Plain shuffled k-fold could give a fold with 0 positives, making recall undefined.

**(c) CV estimate of generalisation error.** 5-fold accuracies $0.82,\ 0.85,\ 0.80,\ 0.88,\ 0.85$.
$\bar s=0.840$; sample std $=0.0308$; standard error $=0.0308/\sqrt5=\mathbf{0.0138}$. Report **accuracy ≈ 84.0 % ± 3.1 %** (SE 1.4 %). Estimated generalisation error $=1-0.84=16\%$.

**(d) Time-series split (3 splits, 6 ordered points).** Train/test = ($\{0,1,2\}$, $\{3\}$), ($\{0..3\}$, $\{4\}$), ($\{0..4\}$, $\{5\}$): the training window grows and the test point is always **in the future**.

---

## N29. Grid-search cost and Group k-fold

**(a) Grid search count.** Grid: $C\in\{0.1,1,10\}$ (3), kernel $\in\{\text{linear},\text{rbf},\text{poly},\text{sigmoid}\}$ (4), class_weight $\in\{\text{None},\text{balanced}\}$ (2) ⇒ $3\times4\times2=\mathbf{24}$ combinations. With 5-fold CV ⇒ $24\times5=\mathbf{120}$ fits (+1 final refit). **Random search** with `n_iter=10` ⇒ $10\times5=50$ fits.
**Nested CV:** 5 outer folds × 3 inner folds × 24 combinations $=\mathbf{360}$ inner fits (+5 refits), the price of an unbiased estimate.

**(b) GroupKFold.** 10 samples from 4 patients: groups $A,A,A,B,B,C,C,C,D,D$. `GroupKFold(2)`: fold 1 trains on $\{A,B\}$ and tests on $\{C,D\}$; fold 2 trains on $\{C,D\}$ and tests on $\{A,B\}$ (**no patient in both train and test**). **Leave-One-Group-Out** gives 4 splits: test on A (train B,C,D), test on B, test on C, test on D. Ordinary k-fold could put patient A's rows on both sides ⇒ leakage and an over-optimistic score.
