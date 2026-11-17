# MACSE504: Machine Learning - Module 3
# Comprehensive Study Notes: Reinforcement Learning (RL)

<div class="meta-box">
<strong>Course:</strong> MACSE504 Machine Learning (ETH 1.0) | School of Computer Science and Engineering<br>
<strong>Prescribed Textbooks:</strong><br>
1. R. S. Sutton and A. G. Barto, <em>Reinforcement Learning: An Introduction</em>, 2nd ed., MIT Press.<br>
2. L. Goel, <em>Artificial Intelligence: Concepts and Applications</em>, Wiley, 2021.<br>
3. T. Jo, <em>Machine Learning Foundations</em>, Springer Nature, 2021.<br>
<strong>VIT Exam Target:</strong> 10-Mark Detailed Answers for Bellman Equations, Q-Learning TD Updates, Policy Gradient Theorem, and Dyna-Q Architecture.
</div>

---

## 📺 Curated Video Lectures by Indian Professors (Under 20 Minutes)
* **Gate Smashers (Varun Singla) - Reinforcement Learning Introduction & MDP (15:20 min):** [Watch on YouTube](https://www.youtube.com/watch?v=JgvyzIkgxF0)  
  *Core Focus:* Agent-environment cycle, Markov Decision Process 5-tuple, discount factor $\gamma$.
* **Krish Naik - Q-Learning Algorithm with Complete Math Intuition (18:10 min):** [Watch on YouTube](https://www.youtube.com/watch?v=1b_rQ9Q8e_U)  
  *Core Focus:* Bellman optimality equation, Temporal Difference (TD) error, and Q-table updates.
* **CampusX (Nitish Singh) - Policy Gradient Methods & REINFORCE Algorithm (19:40 min):** [Watch on YouTube](https://www.youtube.com/watch?v=A_2U69502as)  
  *Core Focus:* Log-derivative trick, trajectory return weighting, and baseline subtraction.

---

## 1. Markov Decision Process (MDP) Framework

<div class="diagram-box">
  <img src="images/ml_mdp_cycle.svg" style="max-width:540px;" alt="RL MDP Agent Environment Interaction">
  <div class="diagram-caption">Figure 3.1: Reinforcement Learning Agent-Environment interaction loop and Bellman backup.</div>
</div>

A Markov Decision Process is formally defined as a 5-tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:
1. $\mathcal{S}$: Set of all valid environment states.
2. $\mathcal{A}$: Set of all agent actions.
3. $\mathcal{P}(s' | s, a) = P(S_{t+1} = s' | S_t = s, A_t = a)$: State transition probability function satisfying the **Markov Property**:
   $$P(S_{t+1}=s' | S_t=s, A_t=a, S_{t-1}, A_{t-1}, \dots) = P(S_{t+1}=s' | S_t=s, A_t=a)$$
4. $\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']$: Expected immediate scalar reward.
5. $\gamma \in [0, 1)$: Discount factor ensuring convergence of infinite-horizon returns:
   $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}$$

---

## 2. Value Functions & Bellman Equations

### 2.1 State-Value Function $V^\pi(s)$
Expected cumulative return starting from state $s$ following policy $\pi(a|s)$:
$$V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s'|s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right]$$

### 2.2 Action-Value Function $Q^\pi(s, a)$
Expected return taking action $a$ in state $s$ and thereafter following $\pi$:
$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} \mathcal{P}(s'|s, a) \left[ \mathcal{R}(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s', a') \right]$$

### 2.3 Bellman Optimality Equations
The optimal policy $\pi^*$ achieves maximal value across all states:
$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s'|s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^*(s') \right]$$
$$Q^*(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s'|s, a) \left[ \mathcal{R}(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

---

## 3. Value-Based RL: The Q-Learning Algorithm

Q-Learning is a **model-free, off-policy Temporal Difference (TD)** control algorithm. It learns the optimal action-value function $Q^*$ directly without requiring transition probabilities $\mathcal{P}$.

<div class="formula-box">
<strong>Q-Learning Temporal Difference Update Rule:</strong><br>
Upon observing transition $(s_t, a_t, r_{t+1}, s_{t+1})$:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]}_{\text{TD Error } \delta_t}$$
where:
- $\alpha \in (0, 1]$ is the learning rate.
- $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$ is the <strong>TD Target</strong>.
- It is <em>off-policy</em> because the target uses the greedy action $\max_{a'}$, while action $a_t$ is selected using an exploratory policy (e.g., $\epsilon$-greedy).
</div>

---

## 4. Policy-Based RL: Policy Gradient Theorem & REINFORCE

Instead of deriving policies indirectly via value functions, policy-based methods parameterize the policy directly as $\pi_\theta(a|s) = P(A=a | S=s; \theta)$.

### 4.1 Policy Objective Function
Maximize expected trajectory return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau; \theta) R(\tau) d\tau$$

### 4.2 The Policy Gradient Theorem
Using the log-derivative trick $\nabla_\theta P(\tau;\theta) = P(\tau;\theta) \nabla_\theta \log P(\tau;\theta)$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t | S_t) G_t \right]$$

### 4.3 Advanced Policy Optimization (PPO)
Standard policy gradients suffer from unstable destructive policy updates. **Proximal Policy Optimization (PPO)** enforces a clipped surrogate objective:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
where probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ and $\hat{A}_t$ is the Advantage estimate.

---

## 5. Model-Based RL & The Dyna-Q Architecture

* **Model-Free RL:** Learns solely from real experience $(s, a, r, s')$. High sample complexity, but no model bias.
* **Model-Based RL:** Learns an explicit model of transition dynamics $\hat{\mathcal{P}}(s'|s, a)$ and rewards $\hat{\mathcal{R}}(s, a)$ to simulate rollouts and plan.
* **Dyna-Q Architecture:** Seamlessly integrates real experience learning, model learning, and simulated planning:
  1. Act in real environment and observe $(s, a, r, s')$.
  2. Direct Q-learning update on real transition.
  3. Update learned model: $\text{Model}(s, a) \leftarrow (r, s')$.
  4. Perform $N$ simulated planning steps: Sample previously visited $(s_k, a_k)$ from model, simulate $(r_k, s'_k) = \text{Model}(s_k, a_k)$, and perform Q-learning update.

---

## 6. Step-by-Step Solved 10-Mark Numerical Problem (VIT Format)

<div class="exam-box">
<strong>VIT Exam Problem [10 Marks]:</strong><br>
An agent in an MDP is in state $s_1$ and chooses action $a_1$.<br>
The agent receives immediate reward $r = +5$ and transitions to state $s_2$.<br>
Given parameters: Learning rate $\alpha = 0.5$, Discount factor $\gamma = 0.8$.<br>
The current Q-table values are:<br>
- $Q(s_1, a_1) = 4.0, \quad Q(s_1, a_2) = 2.0$<br>
- $Q(s_2, a_1) = 6.0, \quad Q(s_2, a_2) = 10.0$<br><br>
(a) Compute the Temporal Difference (TD) Target. (3 Marks)<br>
(b) Compute the Temporal Difference (TD) Error $\delta_t$. (3 Marks)<br>
(c) Compute the updated value $Q^{\text{new}}(s_1, a_1)$. (4 Marks)
</div>

#### Step 1: Compute TD Target
$$\text{TD Target} = r + \gamma \max_{a'} Q(s_2, a')$$
Find greedy action-value at next state $s_2$:
$$\max_{a'} Q(s_2, a') = \max(Q(s_2, a_1), Q(s_2, a_2)) = \max(6.0, 10.0) = \mathbf{10.0}$$
Substituting values:
$$\text{TD Target} = 5.0 + 0.8 \times 10.0 = 5.0 + 8.0 = \mathbf{13.0}$$

#### Step 2: Compute TD Error $\delta_t$
$$\delta_t = \text{TD Target} - Q^{\text{current}}(s_1, a_1) = 13.0 - 4.0 = \mathbf{+9.0}$$

#### Step 3: Compute Updated $Q(s_1, a_1)$
$$Q^{\text{new}}(s_1, a_1) = Q^{\text{current}}(s_1, a_1) + \alpha \cdot \delta_t$$
$$Q^{\text{new}}(s_1, a_1) = 4.0 + 0.5 \times (9.0) = 4.0 + 4.5 = \mathbf{8.5}$$

**Final Result:** After this single experience transition, the estimated return $Q(s_1, a_1)$ increases from **4.0 to 8.5**.
