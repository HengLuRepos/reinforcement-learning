# Reinforcement Learning

## 0. Intro



## 1. Multi-Armed Bandits



## 2. MDP, Value Iteration, Policy Iteration



## 3. Value-Based Methods



## 4. Function Approximation



## 5. Policy Gradient Methods

In policy based methods, we try to find the optimized policy directly without knowledge of a value function. Gradient based RL is a hot topic in policy-based methods. It's basically a optimization problem. We try to find parameter $\theta$ that maximizes the policy objective function $U(\theta)$. The general procedure for policy based methods are as follows:

1. Randomly initialize parameters $\theta$
2. Sample trajectories $\{\tau_i = \{s_t^i,a_t^i\}_{t=0}^T \}$ following current policy $\pi_\theta(a_t|s_t)$.
3. Compute gradient $\nabla_\theta U(\theta)$.
4. $\theta \leftarrow \theta + \alpha\nabla_\theta U(\theta)$.
5. Back to 2.

It's worth mention that there exist some alternatives to gradient based optimization, e.g., Hill Climbing, Genetic Algorithms.

Compared with value based methods, the gradient methods works effectively in high-dim/continuous action spaces. They can also learn stochastic policies.

Generally speaking, there are three types of policy functions. 

- deterministic continuous policy: $a = \pi_\theta(s)$.
- stochastic continuous policy: $a\sim N(\mu_\theta(s),\sigma_\theta^2(s))$
- stochastic discrete policy: $a = \pi_\theta(a|s)$. 

### 5.1 Policy Gradient

#### 5.1.1 Monte Carlo Policy Gradient

Considering the trajectory reward: $R(\tau) = \sum_{t=0}^H R(s_t,a_t)$, then a reasonable objective function is given by
$$
U(\theta) = E_{\tau\sim P(\tau;\theta)}[R(\tau)] = \sum_\tau P(\tau;\theta)R(\tau),
$$
then our goal is find $\theta^* = {\rm argmax}_{\theta} U(\theta)$. To achieve this, we need to compute $\nabla_\theta U(\theta)$.
$$
\begin{equation}
\begin{aligned}
\nabla_\theta U(\theta) &= \nabla_\theta\sum_\tau P_\theta(\tau)R(\tau)\\&=
\sum_\tau\nabla_\theta P_\theta(\tau)R(\tau)\\&=
\sum_\tau P(\tau;\theta)\frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)}R(\tau)\\&=
E_{\tau\sim P_\theta(\tau)}[\nabla_\theta {\rm log}(P_\theta(\tau))R(\tau)]\\&\approx
\frac{1}{N}\sum_{i=1}^N\nabla_\theta {\rm log}(P_\theta(\tau^{(i)}))R(\tau^{(i)}),
\end{aligned}
\end{equation}
$$
where
$$
\begin{equation}
\begin{aligned}
\nabla_\theta {\rm log}(P_\theta(\tau^{(i)})) &= \nabla_\theta \log\prod_{t=0}^T P(s_{t+1}^{(i)}|s_t^{(i)},a_t^{(i)})\pi_\theta(a_t^{(i)}|s_t^{(i)})\\&=
\nabla_\theta\sum_{t=0}^T\log P(s_{t+1}^{(i)}|s_t^{(i)},a_t^{(i)}) +\log \pi_\theta(a_t^{(i)}|s_t^{(i)})\\&=
\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_t^{(i)}|s_t^{(i)})
\end{aligned}
\end{equation}
$$
inserting this into last equation, we get
$$
\begin{equation}
\begin{aligned}
\hat{g} = \nabla_\theta U(\theta) &\approx \frac{1}{N}\sum_{i=1}^N\nabla_\theta {\rm log}(\pi_\theta(a_t^{(i)}|s_t^{(i)}))R(\tau^{(i)})\\&=
\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta {\rm log}(\pi_\theta(a_t^{(i)}|s_t^{(i)}))\sum_{k=0}^NR(s_k^{(i)},a_k^{(i)})\\&=
\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta {\rm log}(\pi_\theta(a_t^{(i)}|s_t^{(i)}))\sum_{k=t}^NR(s_k^{(i)},a_k^{(i)})\\&=
\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta {\rm log}(\pi_\theta(a_t^{(i)}|s_t^{(i)}))G_t^{(i)},
\end{aligned}
\end{equation}
$$
In the derivation, we use the causality, which means the actions taken later won't effect previous reward. This gives REINFORCE algorithm.

#### 5.1.2 Baseline

$G_t$ is an unbiased estimator, but it has high variance. To reduce variance, we subtract a baseline $b$ from $\hat{g}$:
$$
\begin{equation}
\begin{aligned}
\hat{g} &= \sum_{\tau}P(\tau;\theta)\log P(\tau;\theta)(R(\tau)-b)\\&=
\sum_{\tau}P(\tau;\theta)\log P(\tau;\theta)R(\tau)
-\sum_\tau P(\tau;\theta)\frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}b\\&=
\sum_{\tau}P(\tau;\theta)\log P(\tau;\theta)R(\tau) - b\nabla_\theta\sum_\tau P(\tau;\theta) \\&=
\sum_{\tau}P(\tau;\theta)\log P(\tau;\theta)R(\tau),
\end{aligned}
\end{equation}
$$
as long as $b$ is only a function of states. A typical choice of $b$ is the value function $V_\phi^\pi(s)$, thus
$$
\hat{g} =\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta {\rm log}(\pi_\theta(a_t^{(i)}|s_t^{(i)}))\left(G_t^{(i)}-V_\phi^\pi(s_t^{(i)})\right),
$$
where $G_t - V(s)$ is called the *advantage*.

To find the baseline, we can use either MC or TD.

- Monte Carlo approach:
  1. initialize $\phi$.
  2. collect trajectories: $\tau_1,\cdots, \tau_N$.
  3. $\phi \leftarrow \arg\min_\phi \sum_{i=1}^N\sum_{t=0}^{T-1}\left(V_\phi^\pi(s_t^{(i)}-\sum_{k=t}^{T-1}R(s_k^{(i)},a_k^{(i)}\right)^2$.
- TD estimate:
  1. initialize $\phi$.
  2. collect data $s,a,r,s'$.
  3. Fitted $V$ iteration: $\phi_{l+1}\leftarrow \arg\min_\phi\sum_{(s,a,r,s')}||(r+V_{\phi_l}^\pi(s'))-V_\phi(s)||^2$.

### 5.2 Actor Critic Methods

The actor critic methods maintain two set of parameters. 

- Critic updates state-action value function parameters $w$
- Actor updates policy parameters $\theta$, in direction suggested by critic.

The workflow of actor critic:

1. initialize policy parameter $\theta$ and critic parameter $\phi$.
2. sample trajectories under the current policy $\pi_\theta(a_t|s_t)$.
3. Fit value function $V_\phi^\pi(s)$ by MC or TD, update $\phi$.
4. Compute action advantages $A^\pi(s_t^i,a_t^i) = G_t^{(i)} - V_\phi^\pi(s_t^{(i)})$.
5. $\nabla_\theta U(\theta) \approx \hat{g} = \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta {\rm log}(\pi_\theta(a_t^{(i)}|s_t^{(i)}))A^\pi(s_t^{(i)},a_t^{(i)})$.
6. $\theta \leftarrow + \alpha \nabla_\theta U(\theta)$.

In step 4, the $G_t$ is an unbiased estimate of the $Q$ value. If we use bootstrap(like TD learning), $A^\pi(s_t^i,a_t^i) = R(s_t^{(i)},a_t^{(i)}) + \gamma V_\phi^\pi(s_{t+1}^{(i)}) - V_\phi^\pi(s_t^{(i)},a_t^{(i)})$, We get the **Advantage Actor Critic**(A2C) method.

Since trajectories are sampled from current policy, this AC is on policy. We can get off-policy AC by using importance sampling. 

### 5.3 Natural Policy Gradients

We've figured out how to compute gradients, but we haven't found out how to decide the stepsize. If the stepsize is too large, we may run into bad policies which we may never recover from. Thus the stepsize must be constrained. 

- **Natural gradient descent**: the step is determined by:
  $$
  d^* = \arg\max U(\theta+d), D_{KL}(\pi_\theta||\pi_{\theta+d})\leq \epsilon,
  $$
  which is 
  $$
  d^* =\arg\max_d \nabla_\theta U(\theta)|_{\theta=\theta_{old}} - \frac{1}{2}\lambda(d^TF(\theta_{old})d).
  $$

- **Trust Region Policy Optimization**: NPG+LinearSearch+monotonic improvement theorem

- **Proximal Policy Optimization**: Clipped objective
  $$
  L_{\theta_k}(\theta) = E_\pi\left[\sum_{t=0}^T\left[\min(r_t(\theta)\hat{A}_t,{\rm clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right]\right], r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}
  $$

### 5.4 Deterministic PG, Re-parametrized PG

Consider another policy objective:
$$
\max_\theta E_\tau\left[\sum_{t=1}^T Q(s_t,a_t)\right].
$$
The problem is how to backpropagate? If the policy is deterministic, i.e. $a = \pi_\theta(s)$,
$$
E\sum_t\frac{dQ(s_t,a_t)}{d\theta} = E\sum_t\frac{dQ}{da_t}\frac{da_t}{d\theta}
$$

<img src=".\pictures\DDPG.png" alt="DDPG" style="zoom:50%;" />

### 5.5 Maximum Entropy RL

Standard RL tries to maximize the expected sum of rewards
$$
J(\pi) = \sum_{t=1}^T E_{(s_t,a_t)\sim \rho_\pi}R(s_t,a_t),
$$
where $\rho_{\pi}$ is the state-action marginal probabilities induced from $\pi$, and the optimal policy is given by
$$
\pi^* = \arg\max_\pi J(\pi).
$$
The optimal policy is a deterministic policy. However, in some cases we might prefer stochastic policies. Stochastic policies are useful because:

- learn all possible ways to accomplishing a task
- Better policies to finetune from for specific env
- Better exploration
- **Robust for hyperparameters**

Thus we need a objective function that favors stochastic policies. The maximum entropy RL objective function is given by
$$
J(\pi) = \sum_{t=1}^T R(s_t,a_t) + \alpha H(\pi(\cdot|s_t)),
$$
where 
$$
H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)
$$
is the entropy and $\alpha$ is used to determine the relative importance of the reward and entropy. 

For discrete action sets, instead of using argmax, we use softmax over estimated Q values and sample actions from it
$$
\pi(a_t|s_t)\propto \exp\left(-\frac{1}{\alpha}Q^{soft}(s_t,a_t)\right),
$$
where $Q^{soft}_\pi$ is expected cumulative sum of rewards + expected cumulative sum of entropy.

**Soft Bellman equation**:
$$
q_\pi^{soft}(s,a) = r(s,a) + \sum_{s'\in S}T(s'|s,a)\sum_{a'\in A}(q^{soft}_\pi(s',a')-\alpha H(\pi(a'|s')))
$$
**Soft Bellman backup update operator**:
$$
Q^{soft}(s_t,a_t) \leftarrow r(s_t,a_t) + \gamma E_{s_{t+1},a_{t+1}}\left[Q^{soft}(s_{t+1},a_{t+1}) -\alpha H(\pi(a_{t+1}|s_{t+1})) \right]
$$
define
$$
r^{soft}(s_t,a_t) = r(s_t,a_t) + \gamma \alpha E_{s_{t+1}\sim \rho}H(\pi(\cdot|s_{t+1}))
$$
then
$$
Q^{soft}(s_t,a_t) \leftarrow r^{soft}(s_t,a_t) + \gamma E_{s_{t+1},a_{t+1}}\left[Q^{soft}(s_{t+1},a_{t+1})\right]
$$

#### 5.5.1 Soft Policy Iteration(Tabular case)

Soft policy iteration iterates between two steps:

1. *Soft policy evaluation*: till convergence
   $$
   Q^{soft}(s,a) \leftarrow r(s,a) + \gamma E_{s',a'}\left[Q^{soft}(s',a') -\alpha H(\pi(a'|s')) \right]
   $$
   This converges to $Q^\pi$.

2. *Soft policy improvement*: Update policy
   $$
   \pi_{new} = \arg\min_{\pi'\in \Pi}D_{KL}\left[\pi'(\cdot|s_t)||\frac{\exp\frac{1}{\alpha}Q^{old}(s_t,\cdot)}{Z^{old}(s_t)}\right]
   $$

#### 5.5.2 Soft Policy Iteration: Approximation

1. Learning $Q_\theta(s_t)$:
   $$
   J_Q(\theta) = E_{(s_t,a_t)\sim D}\left[\frac{1}{2}\left(Q_\theta(s_t,a_t) - \hat{Q}(s_t,a_t)\right)^2\right]
   $$
   Using semi-gradient:
   $$
   \nabla_\theta J_Q(\theta) = \nabla_\theta Q_\theta(s_t,a_t)\left(Q_\theta(s_t,a_t)-(r(s_t,a_t)+\gamma\left( Q_{\tilde{\theta}}(s_{t+1},a_{t+1})-\alpha \log\pi_{\phi}(a_{t+1}|s_{t+1}))\right) \right)
   $$
   Here we use TD target

2. Learning $\pi_\phi(a_t|s_t)$:
   $$
   J_\pi(\phi) = E_{s_t\in D}D_{KL}\left[\pi_\phi(\cdot|s_t)||\frac{\exp\frac{1}{\alpha}Q_{\theta}(s_t,\cdot)}{Z_{\theta}(s_t)}\right]
   $$
   and the gradient
   $$
   \nabla_\phi J_\pi(\phi) = \nabla_\phi E_{s_t\in D}E_{a_t\sim \pi_\phi(a|s_t)}\left[\log\pi_\phi(a_t|s_t)-\frac{1}{\alpha}Q_\theta(s_t,a_t) \right]
   $$
   Re-parameterization: $a_t = f_\phi(s_t,\epsilon) = \mu_\phi(s_t)+\epsilon \Sigma_\phi(s_t), \epsilon\sim N(0,1)$
   $$
   \nabla_\phi J_\pi(\phi) =  E_{s_t\in D, \epsilon\sim N(0,1)}\nabla_{a_t}\left[\log\pi_\phi(a_t|s_t)-\frac{1}{\alpha}Q_\theta(s_t,a_t) \right]\nabla_\phi a_t
   $$

#### 5.5.3 Composability of Maximum Entropy Policies



#### 5.5.4 Soft Actor Critic



## 6. Evolutional Methods for policy search

General evolutional algorithm:

0. Initialize a population of parameter vectors
1. Make random perturbations (mutations) to each parameter vector
2. Evaluate the perturbed parameter vector(fitness)
3. Keep the perturbed vector if the result improves(selection)
4. GOTO 1

### 6.1 Cross-Entropy Method(CEM)

- Policy params are sampled from a multivariate Gaussian distribution with a **diagonal covariance matrix**
- update mean and variances of the parameter using samples that have highest fitness scores(elites)

<img src=".\pictures\CEM.png" alt="CEM" style="zoom:50%;" />

### 6.2 Covariance Matrix Adaptation(CMA-ES)

- Policy params are sampled from a multivariate Gaussian distribution with a **full covariance matrix**.
- update mean and vars using elites

$$
\mu_{t+1} = \sum_{i=1}^{n_{elit}}w_i\theta_i^{elite,t}
$$

$$
\Sigma_{t+1} = Cov(\theta_1^{elite,t},\theta_2^{elite,t},\cdots) + \epsilon I
$$

### 6.3 Natural Evolutionary Strategies (NES)

- NES considers **every** offspring
- $\theta \sim P_\mu(\theta)$, $P_\mu(\theta)$ is a Gaussian distribution with mean $\mu \in \R^d$ and **diagonal and fixed covariance matrix** $\sigma^2I_d$.
- Goal: $\max_\mu E_{\theta\sim P_\mu(\theta)}F(\theta)$, where fitness score $F(\theta) = E_{\tau\sim \pi_\theta,s_0\sim \mu_0(s)} R(\tau)$.

$$
\begin{equation}
\begin{aligned}
\nabla_\mu E_{\theta\sim P_\mu(\theta)}F(\theta) &= \nabla_\mu\int P_\mu(\theta) F(\theta)\ d\theta\\&=
\int\nabla_\mu P_\mu(\theta) F(\theta)\ d\theta\\&=
\int P_\mu(\theta)\nabla_\mu \log P_\mu(\theta) F(\theta)\ d\theta\\&=
E_{\theta\sim P_\mu(\theta)}[\nabla_\mu \log P_\mu(\theta) F(\theta)]\\&\approx
\frac{1}{N}\sum_{i=1}^N\left[\nabla_\mu \log P_\mu(\theta)|_{\theta = \theta_i} F(\theta_i) \right],\ \theta_i\sim P_\mu(\theta)
\end{aligned}
\end{equation}
$$

Sampling from a General Gaussian denstity:
$$
\mathbf x\sim N(\mu,\Sigma), \ x\sim \mu + L\mathscr{N}(0,1), \Sigma = LL^T
$$
<img src=".\pictures\NES.png" alt="NES" style="zoom:50%;" />

### 6.4 Scalability of Evolutionary Methods

Evolutionary methods work well on low dimensional problems. Can they be used to optimize policies represented as neural networks of thousands parameters?

#### Distributed Stochastic Gradient Descent

Every worker get a set of examples, and computes a gradient vector. The master averages those gradient vectors.

<img src="D:.\pictures\DSGD.png" alt="DSGD" style="zoom:50%;" />

<img src=".\pictures\PES.png" alt="PES" style="zoom:50%;" />

However, ES can get stuck in local optima. A empirical suggestion is that searching for policies in multiple related tasks and in multiple related envs can help ES methods work better.



## 7. Model-Based RL

In previous sections, we mainly focus on model-free reinforcement learning, i.e., MC, TD, policy gradient. These methods don't involve the model of the environment, which is the state transition $T(s'|s,a)$ plus the reward function $R(s,a)$. But models can be useful. If model is learnt, action selection can be performed by lookahead(or model forward unrolling) during training or testing. Also, real experience can be augmented by synthetic experience generation. This gives rise to the need for model-based RL.

MBRL methods can learn a model and use it to select actions and learn policies. MBRL methods typically have more experience efficiency and the learned models can support different tasks.

A question naturally arise: how to learn models. One might attempt to randomly initialize the model, sample from the model, use returns to optimize the model and repeat, which seems pretty close to the policy iteration. But this method is problematic. The model is trained by the sampled data, thus its accuracy is only guaranteed on these training data. And if we plan through the learned model, the sampled distribution may differ greatly with the training data distribution. Thus we need the policy learning to perform action selection. It works as follows:

<!--TODO: put Alternating here -->

However, model leaning is unfortunately challenging:

- Under-modeling

- Over-fitting

- Errors compound through unrolling(which can be relieved by model-predictive control)

- Need to capture stochasticity of the environment

- **Disconnect between model learning objectives and reward optimization**.

  - Model learning(training) obj: 
    $$
    \arg\max_{\theta}\sum_i \log p_\theta(s_i'|s_i,a_i)
    $$
    Reward optimization(Control) obj:
    $$
    \arg\max_{a_{t:t+T}}E_{\pi_{\theta}s(t)}\sum_{i=t}^{t+T}r(s_i,a_i)
    $$

  - Models are trained for getting good policies(and thus good rewards), but they are actually trained to maximize lowlihood of transitions. 



### 7.1 MBRL in low dimensional space

<!--TODO: HyperGAN -->

Generally, to estimate optimal policy in MBRL, we can either finetune the policy using model-free method, or imitate a model-based controller(DAGGER). However, these two methods are relatively computational expensive. Researchers from UCB proposed a method called **Probabilistic Ensembles Trajectory Sampling**(PETS) which outperforms model-free methods while requiring significantly fewer samples.

> We propose a new algorithm called probabilistic ensembles with trajectory sampling (PETS) that combines uncertainty-aware deep network dynamics models with sampling-based uncertainty propagation

The models generally have two types of uncertainty:

- Epistemic uncertainty: uncertainty due to lack of data
- Aleatoric uncertainty: uncertainty due to the inherent stochasticity of the system

To represent Aleatoric uncertainty, instead of outputting a single state given current state and action, the model outputs a *Gaussian distribution* over next state, and a neural network is used to predict mean and covariance matrix for a given $(s,a)$ pair.

To present Epistemic uncertainty, we exploit the posterior distribution from Bayes statistics. However, the inference of such distribution is intractable. 

- Neural network ensembles are a good estimation to Bayesian nets.
- We have a bunch of NNs, each trained on separate data(train with different initializations and different subset).

PETS algorithm works as follows:

<!--TODO: PETS -->

Model-ensemble trust region policy optimization:

<!-- TODO: ensemble TRPO-->

Model-Based Policy Optimization:

<!-- TODO: MBPO-->

### 7.2 AlphaGo, AlphaGoZero

<!--Skip-->

Monte Carlo search tree for Go is too wide and deep. 

AlphaGo

### 7.3 MBRL in sensory space

Sometimes we only have access to sensory inputs, such as videos or tactile signals. There're generally two ways to perform model learning from sensory inputs:

- unrolling in the observation space: the model is trained to predict future *observations* (e.g. pixels of future frames)
- unrolling in the latent space: we may use observation reconstruction as an auxiliary loss.

We mainly focus on the latent space unrolling. 

### 7.4 MBRL Deterministic latent dynamics models (muZero)

<!--Skip-->

### 7.5 MBRL Stochastic latent dynamics models

$s_{t+1}\sim p(s_{t+1}|s_t,a_t)$.

<img src=".\pictures\DREAMER_ILLUST.png" alt="DREAMER_ILLUST" style="zoom:50%;" />

<img src=".\pictures\DREAMER_ALGO.png" alt="DREAMER_ALGO" style="zoom:50%;" />



## 8. Curiosity-driven Exploration



## 9. Imitation Learning



### 9.1 kinesthetic imitation



### 9.2 visual imitation

