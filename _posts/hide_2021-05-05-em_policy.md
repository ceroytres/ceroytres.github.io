# Policy Iteration as Expectation Maximization

## Under construction!!

## Expectation Maximization (EM)

EM is an algorithm for finding the maximum likelihood parameters $\theta$ for probabilistic models having latent variables. EM consists of two steps E (Expectation) and M (Maximization). EM is a special case of a [MM algorithm](https://en.wikipedia.org/wiki/MM_algorithm) for optimization.

Suppose we want to find the maximum likelihood for a probablistic model with latent variables. The set of observed variables is given denote by $\mathbf{x}$ while the set of latent variables is given by $\mathbf{z}$. Suppose the joint distribution is parameterized by $\theta$, $p(\mathbf{x}, \mathbf{z}\|\theta)$. Suppose we need to maximize the likelihood of the observed data; therefore, we seek find the parameters $\theta$ that maximize the likelihood of the marignal distribution $p(\mathbf{x}\|\theta)$. Although optional, it is common practice to use the log-likelihood.
The log likelihood function of the observed variables is given by 

$$\log p(\mathbf{x}|\theta) = \log \sum_{\mathbf{z}} p(\mathbf{x}, \mathbf{z}|\theta)$$

The underlying assumpition is that $p(\mathbf{x},\mathbf{z}\|\theta)$ 
yields an easier optimization problem then $p(\mathbf{x}\|\theta)$. For any choice of distribution $q$ over $\mathbf{z}$, the following holds:

$$\log p(\mathbf{x}|\theta) = \mathcal{L}(q,\theta) + \text{KL}(q||p)$$

where 

$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}} q(\mathbf{z}) \log \frac{p(\mathbf{x},\mathbf{z}|\theta)}{q(\mathbf{z})}$$

$$ \text{KL}(q||p) = -\sum_{\mathbf{z}} q(\mathbf{z}) \log \frac{p(\mathbf{z}|\mathbf{x},\theta)}{q(\mathbf{z})}$$

<details>
  <summary>Proof</summary>
  
$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}} q(\mathbf{z}) [\log p(\mathbf{x},\mathbf{z}|\theta) - \log q(\mathbf{z})]$$
$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}} q(\mathbf{z}) [\log p(\mathbf{z}|\mathbf{x},\theta) + \log p(\mathbf{x}) - \log q(\mathbf{z})]$$
$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}} q(\mathbf{z}) [\log p(\mathbf{z}|\mathbf{x},\theta) - \log q(\mathbf{z})] + \log p(\mathbf{x}) $$

$$ \mathcal{L}(q, \theta) = -\text{KL}(q||p) + \log p(\mathbf{x}) $$ 

</details>

Since $\text{KL}(q\|\|p) \ge 0$, then it follows that $\mathcal{L}(q,\theta) \le \log p(\mathbf{x})$. Therefore $\mathcal{L}(q,\theta)$ is a lower bound to $\log p(\mathbf{x})$. 

Suppose the current parameter are denoted by $\theta^{\text{old}}$. In E-step, the lower bound $\mathcal{L}(q,\theta^\text{old})$ is maximized with respect to $q$ with $\theta^\text{old}$ fixed. The tightest lower bound $\mathcal{L}(q,\theta^\text{old})$ occurs when $q(\mathbf{z}) = p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$ since the KL divergence term disappears and as a result, $\mathcal{L}(q,\theta^\text{old}) = \log p(\mathbf{x} \| \theta^{\text{old}})$. Note for $q(\mathbf{z}) = p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$, $\mathcal{L}(q, \theta) \le \log p(\mathbf{x}\| \theta)$ for $\theta \neq \theta^{\text{old}}$ but achieves equality for $\theta = \theta^\text{old}$. $\mathcal{L}(p(\mathbf{z}\|\mathbf{x},\theta^\text{old}), \theta)$ can be seen a minorizing surrogate function.

In the M-step, the distribution $q(\mathbf{z}) = p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$  is held fixed but the lower bound is maximized with respect to the parameters $\theta$. The updated parameters are denoted by $\theta^\text{new}$. Recall that $\mathcal{L}(q, \theta^{\text{old}}) = \log p(\mathbf{x} \| \theta^{\text{old}})$, $\log p(\mathbf{x} \| \theta^{\text{old}}) \le \mathcal{L}(q, \theta^\text{new})$. Since $q(\mathbf{z}) = p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$, $\mathcal{L}(q, \theta^\text{new}) \le \log p(\mathbf{x}\|\theta^\text{new})$ and therefore $\log p(\mathbf{x} \| \theta^{\text{old}}) \le \log p(\mathbf{x} \| \theta^{\text{new}})$. In next E-step, $q(\mathbf{z})$ will be set to $p(\mathbf{z}\|\mathbf{x},\theta^\text{new})$ thus re-establishing equality, and in the following M-step, the new parameters will increase or maintain the same likelihood.

In the E-step, if we set $q(\mathbf{z}) = p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$

$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}}p(\mathbf{z}|\mathbf{x},\theta^\text{old}) \log \frac{p(\mathbf{x},\mathbf{z}|\theta)}{p(\mathbf{z}|\mathbf{x},\theta^\text{old})}$$

$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}}p(\mathbf{z}|\mathbf{x},\theta^\text{old}) [\log p(\mathbf{x},\mathbf{z}|\theta) - \log p(\mathbf{z}|\mathbf{x},\theta^\text{old})]$$

Since the second term does not rely on $\theta$ during the M-step it is considered a constant and as a result, the M-step effectively optimizes $Q$:

$$ \mathcal{L}(q, \theta) = \sum_{\mathbf{z}}p(\mathbf{z}|\mathbf{x},\theta^\text{old}) \log p(\mathbf{x},\mathbf{z}|\theta) + \text
{constant}$$

$$Q(\theta, \theta^{\text{old}}) = \sum_{\mathbf{z}}p(\mathbf{z}|\mathbf{x},\theta^\text{old}) \log p(\mathbf{x},\mathbf{z}|\theta) = \mathbb{E}_{p(\mathbf{z}|\mathbf{x},\theta^\text{old})}[\log p(\mathbf{x},\mathbf{z}|\theta) ]$$

The quantity in the expectation is the log likelihood of the complete distribution, and in order to evaluate $Q$ we need to evaluate $p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$. Typically the primary focus of the E-step is to evaluate $p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$.

**EM Algorithm:**

1. Initialize $\theta^{\text{old}}$

2. **E-Step:** Evaluate $p(\mathbf{z}\|\mathbf{x},\theta^\text{old})$.

3. **M-Step:** $\theta^{\text{new}} = \text{argmax}_\theta Q(\theta, \theta^{\text{old}})$

4. If not converged, $\theta^{\text{old}} \gets \theta^{\text{new}}$ and go to step 2.





## Markov Decision Process

A Markov Decision Process (MDP) consist of the following [3,4]:
* Initial State Distribution $p(s_0) = P(S_0=s_0)$
* State Transition Distribution $p(s\'\|s,a) = P(S_t = s\'\|S_{t-1}=s, A_{t-1}=a)$
* Expected Reward Function $r(s,a) = \sum_r r p(r\|s,a)  = \mathbb{E}[R_t\|S_{t-1},A_{t-1}]$ 
* Policy $\pi(s\|a) = P(S_{t-1}=s\|A_{t-1}=a)$
* Discounting Factor $\gamma \in [0,1)$


Let $\tau$ denote a trajectory of action and states $(S_0, A_0, S_1, A_1, ..., S_T, A_T)$ under the MDP, and it follows the following joint distribution

$$p(\tau) = p(s_0) [\prod_{t=0}^{T-1}  \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)] \pi(a_T|S_T) $$

Solving an MDP consists of finding the policy distribution that maximizes the expected discounted return

$$\mathbb{E}[G_t] = \mathbb{E}[\sum_{t=0}^{\infty}\gamma^k R_{t+k+1}] = \sum_t \gamma^t \mathbb{E}[r(S_t,A_t)]$$

<details>
  <summary>Proof</summary>

For a fixed $t = t'$

$$\mathbb{E}[R_{t'}] = \sum_{r} p(r) r$$

$$ =\mathbb{E}[R_{t'}] = \sum_{r,\tau} p(r,\tau) r$$

$$= \sum_{r,\tau} p(r|\tau) p(\tau) r $$

By Markov Property, $p(r|\tau) = p(r|s,a)$

$$= \sum_{\tau} p(\tau) \sum_r rp(r|s,a) $$

$$= \sum_{\tau} p(\tau) r(s,a) $$

$$= \mathbb{E}[r(S,A)] $$

</details>

The value of a state is given (value function):

$$v_\pi(s) = \mathbb{E}[G_t | S_t = s] $$

The value function satisfies the following recursion relationship:

$$v_\pi(s) = \sum_{a} \pi(a|s) (r(s,a) + \gamma \sum_{s'}p(s'|s,a)v_\pi(s))$$

as known as the Bellman equation for $v_\pi(s)$.

Suppose we also restrict the MPDs to recieve a binary reward $R = \\{0,1\\}$ at a stopping time $T$ and we refer to these as restricted MDPs. One possible interpretation of the discounting factor is probability of stoping at time $T$. The probability of stopping $P(T=t) \propto \gamma^t$. The joint distribution of the restricted MPD and the stopping time 

$$p(\tau, T) = p(\tau| T)p(T) = p(s_0) [\prod_{t=0}^{T-1}  \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)] \pi(a_T|S_T) (1-\gamma)\gamma^T$$


Suppose the reward received at time $T$ followed a distribution such that $P(R=1\|s,a) =\frac{r(s,a) - min(r)}{max(r) - min(r)}$. Therefore the joint distribution of the reward, tajectory, and stopping time: 

$$ p(R, \tau, T) = p(R,\tau| T)p(T) =  p(R|\tau, T)p(\tau| T)p(T) = p(R|S_T,A_T)p(\tau| T)p(T) $$

The distribution $p(R, \tau)$ can be seen as a mixture of these restricted MDPs.

### Policy Iteration as EM

Suppose we want to find a $\theta = \pi$ that maximizes the marginal likelihood of $p(R)$:

$$p(R) = \sum_{\tau, T} p(R, \tau, T) =\sum_{\tau, T} p(R|S_T,A_T)p(\tau| T)p(T) $$

the latent variables consist of $\mathbf{z}=\\{S_t, A_t, t\\}_{t=0}^T$

$$\sum_{\tau, T} p(R|S_T,A_T)p(\tau| T)p(T)  = (1-\gamma) \sum_{T,\tau} \gamma^T p(R|S_T,A_T)p(\tau| T) $$

$$ = (1-\gamma)\sum_{T,\tau} \gamma^T p(s_0) [\prod_{t=0}^{T-1}  \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)] \pi(a_T|S_T) p(R|S_T,A_T)$$

Recall $p(R\|S_T,A_T) \propto r(S_T, A_T)$

$$ \propto \sum_{\tau,T} \gamma^T p(s_0) [\prod_{t=0}^{T-1}  \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)] \pi(a_T|S_T) r(S_T, A_T) $$

For a fixed $T = T\'$

$$\sum_{\tau} \gamma^{T'} p(s_0) [\prod_{t=0}^{T'-1}  \pi(a_t|s_t) p(s_{t+1}|s_t, a_t)] \pi(a_{T'}|s_{T'}) r(S_T', A_T')$$

$$ = \gamma^{T'}\mathbb{E}[r(S_{T'}, A_{T'})] $$

$$ p(R) \propto \sum_{T}\gamma^T \mathbb{E}[r(S_{T}, A_{T})] $$

Therefore maximizing the marginal distribution of the binary reward on a mixture of restricted MDPs is equal to the maximizing the expected return on the original MDP.

See for Section 1.2.1 and 1.3.1 in [1] for the derivation of $Q(\pi, \pi^{\text{old}})$

$$Q(\pi, \pi^{\text{old}}) = \sum_{a,s}[(p(R|s,a) + \sum_{s'}\beta(s')p(s'|a,s))\pi^{\text{old}}(a|s)\log\pi(a|s)] \alpha(s) $$

$\beta(s) \propto v_{\pi^{\text{old}}}(s)$ and $\alpha(s)$ is the probability of state $s$ and only depend on $\pi^{\text{old}}$.

$$Q(\pi, \pi^{\text{old}}) \propto \sum_{a,s}(r(s,a) + \sum_{s'}v_{\pi^{\text{old}}}
(s')p(s'|a,s))\pi^{\text{old}}(a|s)\log\pi(a|s) \alpha(s) $$

Therefore the E-step requires evaluting the value function $v_{\pi^{\text{old}}}(s)$. 

### References

[1] Toussaint, Marc, Amos Storkey, and Stefan Harmeling. "Expectation-Maximization methods for solving (PO) MDPs and optimal control problems." Inference and Learning in Dynamic Models (2010).

[2] Bishop, Christopher M. Pattern Recognition and Machine Learning. Springer, 2006.

[3] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

[4] [David Silver's RL course](https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf)