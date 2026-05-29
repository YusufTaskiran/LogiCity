# Method

## 4.1 Neural RL Baseline

We use Proximal Policy Optimization (PPO) as our fully neural baseline. The policy receives the grounded symbolic observation vector described in Chapter 3 and outputs a categorical distribution over the four macro-actions
\[
\mathcal{A}=\{\texttt{Slow},\texttt{Normal},\texttt{Fast},\texttt{Stop}\}.
\]
In our implementation, the policy is parameterized by a multilayer perceptron feature extractor with layer widths \(d_{\text{obs}} \rightarrow 128 \rightarrow 64 \rightarrow 64\), followed by standard actor and critic heads.

Let \(\pi_\theta(a\mid s)\) denote the policy and \(V_\theta(s)\) the value function. PPO updates the policy by optimizing the clipped surrogate objective. For each sampled transition \((s_t,a_t)\), we define the probability ratio
\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]
Given advantage estimates \(\hat A_t\), the PPO surrogate is
\[
L_{\mathrm{clip}}(\theta)
=
\mathbb{E}_t\!\left[
\min\!\Big(
r_t(\theta)\hat A_t,\;
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\Big)
\right].
\]
Because the implementation is written as a loss minimization problem, the policy loss is
\[
L_{\mathrm{policy}}(\theta)=-L_{\mathrm{clip}}(\theta).
\]
The value loss is
\[
L_{\mathrm{value}}(\theta)=\mathbb{E}_t\!\left[(V_\theta(s_t)-\hat R_t)^2\right],
\]
and the entropy term is included through
\[
L_{\mathrm{entropy}}(\theta)=-\mathbb{E}_t\!\left[\mathcal{H}(\pi_\theta(\cdot\mid s_t))\right].
\]
The total PPO loss minimized in our baseline is therefore
\[
L_{\mathrm{PPO}}(\theta)
=
L_{\mathrm{policy}}(\theta)
+ c_v L_{\mathrm{value}}(\theta)
+ c_e L_{\mathrm{entropy}}(\theta).
\]
The policy parameters are updated by gradient descent,
\[
\theta \leftarrow \theta - \eta \nabla_\theta L_{\mathrm{PPO}}(\theta),
\]
where \(\eta\) denotes the learning rate. In this baseline, learning is driven entirely by task reward; there is no additional symbolic safety mechanism beyond what is already implicit in the observation representation.

## 4.2 LogiCity Probabilistic Logic Shield

Our shielded method instantiates the Probabilistic Logic Shield idea from PLPG in LogiCity using ProbLog. As in the original formulation, the shield consists of three components.

The first component is the policy representation \(\Pi_s\), encoded as an annotated disjunction inside the logic program. For a given state \(s\), we represent the neural policy as
\[
\Pi_s =
\{
\pi_\theta(\texttt{slow}\mid s)::\texttt{act(slow)};
\pi_\theta(\texttt{normal}\mid s)::\texttt{act(normal)};
\pi_\theta(\texttt{fast}\mid s)::\texttt{act(fast)};
\pi_\theta(\texttt{stop}\mid s)::\texttt{act(stop)}
\}.
\]
This is the LogiCity counterpart of the policy component in PLPG.

The second component is the probabilistic fact set \(H_s\), which abstracts the current state into shield-relevant symbolic information. In our setting, these facts are derived directly from the grounded observation vector. For the easy template, the abstraction uses facts such as whether the ego vehicle is at an intersection, whether another entity is in the intersection, whether another entity has higher priority, and whether another entity is on a near-collision course. The medium and hard templates extend this abstraction with richer semantic and spatial facts, including ambulance, bus, police, pedestrian, old, close, left-of, right-of, and next-to relations. Each fact is assigned a probability in \([0,1]\), either deterministically in the perfect-sensor case or stochastically in the noisy-sensor case.

The third component is the background knowledge \(\mathcal{BK}\), which encodes the safety logic. In our LogiCity case, this background knowledge is given by difficulty-specific ProbLog clauses describing hazard conditions and action appropriateness. These clauses are not detached from the simulator. Instead, they are aligned with the task rules used in LogiCity and, in the current implementation, can be translated from the rule YAML files through difficulty-specific translators.

Together, the three components define one ProbLog program per state,
\[
\mathcal{T}(s)=\mathcal{BK}\cup H_s\cup \Pi_s.
\]
We compile the template program once using the `ddnnf` backend in ProbLog and then reuse the compiled circuit at runtime by updating only the fact weights and action weights.

From this program, we compute three quantities. First, the action-level safety score for action \(a\) is obtained by replacing the policy annotated disjunction with a one-hot choice of \(a\) and querying the program for `safe`:
\[
q(s,a)=P_{\mathcal{T}_a(s)}(\texttt{safe}),
\]
where \(\mathcal{T}_a(s)\) denotes the same program with the action distribution collapsed to action \(a\). In the implementation, this is done by evaluating the compiled circuit once per action.

Second, we compute the safety of the base policy,
\[
P_{\pi_\theta}(\texttt{safe}\mid s)
=
\sum_{a\in\mathcal{A}} \pi_\theta(a\mid s)\, q(s,a).
\]
Third, we construct the shielded policy by reweighting the base policy with the action-level safety scores,
\[
\pi_\theta^+(a\mid s)
=
\frac{\pi_\theta(a\mid s)\, q(s,a)}
{\sum_{a'\in\mathcal{A}} \pi_\theta(a'\mid s)\, q(s,a')}
=
\frac{\pi_\theta(a\mid s)\, q(s,a)}
{P_{\pi_\theta}(\texttt{safe}\mid s)}.
\]
This is the policy that is actually used for acting. The safety of the shielded policy is
\[
P_{\pi_\theta^+}(\texttt{safe}\mid s)
=
\sum_{a\in\mathcal{A}} \pi_\theta^+(a\mid s)\, q(s,a).
\]

Our PLPG implementation uses the shielded policy both for rollout collection and for the PPO update. The PPO ratio is therefore formed from shielded log-probabilities, not from the raw base-policy probabilities. In addition, we include an explicit safety regularizer. The total PLPG loss minimized in the implementation is
\[
L_{\mathrm{PLPG}}(\theta)
=
L_{\mathrm{policy}}^{+}(\theta)
+ c_v L_{\mathrm{value}}(\theta)
+ c_e L_{\mathrm{entropy}}^{+}(\theta)
+ \alpha L_{\mathrm{safety}}(\theta),
\]
where \(L_{\mathrm{policy}}^{+}\) and \(L_{\mathrm{entropy}}^{+}\) are computed from the shielded policy \(\pi_\theta^+\), and
\[
L_{\mathrm{safety}}(\theta)
=
-\mathbb{E}_t\!\left[\log P_{\pi_\theta^\star}(\texttt{safe}\mid s_t)\right].
\]
The policy \(\pi_\theta^\star\) can in principle be either the base policy \(\pi_\theta\) or the shielded policy \(\pi_\theta^+\), depending on configuration. In our current setup, the default is the shielded-policy version. As in PPO, the parameter update is performed by gradient descent,
\[
\theta \leftarrow \theta - \eta \nabla_\theta L_{\mathrm{PLPG}}(\theta).
\]
This makes the shield part of the learning computation itself rather than a purely external action filter.

## 4.3 Shield Construction for Safe Path Following

Designing the shield for LogiCity required more than simply translating the task rules into ProbLog. The key issue is that Safe Path Following is not only a collision-avoidance problem. The agent must avoid unsafe interactions, but it must also continue making progress toward its goal. A shield that only suppresses dangerous motion can still fail the task if it induces excessive stopping.

Our first shield design was purely safety-based. In that design, hazard conditions were translated into clauses that marked movement actions as unsafe, while `Stop` remained implicitly safe. At the logic level, this corresponds to semantics of the form
\[
\texttt{unsafe} \leftarrow \texttt{hazard} \land \texttt{act(move)},
\qquad
\texttt{safe} \leftarrow \neg \texttt{unsafe}.
\]
This formulation is natural if shielding is understood only as preventing bad actions. However, in our LogiCity experiments it proved insufficient under uncertainty. When symbolic facts became noisy, the shield penalized movement whenever hazard became even moderately plausible, while `Stop` remained the unique safe fallback. This led to timeout-dominant behavior: the agent avoided many failures, but stopped too often to reach the goal.

For this reason, we redesigned the shield around **appropriateness** rather than pure movement safety. The shield should not only ask whether an action is unsafe, but whether it is appropriate for safe progress in the current state. In the easy template, the hazard conditions are
\[
\texttt{hazard} \leftarrow \texttt{is\_at\_inter\_ego} \land \texttt{is\_in\_inter}_i,
\]
\[
\texttt{hazard} \leftarrow \texttt{is\_at\_inter\_ego} \land \texttt{higher\_pri}_i,
\]
\[
\texttt{hazard} \leftarrow \texttt{colliding\_close}_i,
\]
for relevant surrounding entities \(i\). We then define
\[
\texttt{need\_to\_stop} \leftarrow \texttt{hazard},
\qquad
\texttt{okay\_to\_move} \leftarrow \neg \texttt{need\_to\_stop}.
\]
Action appropriateness is encoded as
\[
\texttt{appropriate} \leftarrow \texttt{act(stop)} \land \texttt{need\_to\_stop},
\]
\[
\texttt{appropriate} \leftarrow \texttt{act(slow)} \land \texttt{okay\_to\_move},
\]
\[
\texttt{appropriate} \leftarrow \texttt{act(normal)} \land \texttt{okay\_to\_move},
\]
\[
\texttt{appropriate} \leftarrow \texttt{act(fast)} \land \texttt{okay\_to\_move}.
\]
The query name remains `safe` in code, but semantically we now define
\[
\texttt{safe} \leftarrow \texttt{appropriate}.
\]

This redesign is central to our method because it aligns the shield with the real task objective. `Stop` is no longer universally preferred whenever uncertainty grows; instead, it is preferred when stopping is actually needed. Conversely, movement remains appropriate when no stop condition is active. The same outer structure is used in the medium and hard templates, but with richer hazard definitions that incorporate additional traffic semantics such as ambulance-related yielding, bus-pedestrian interactions, police-related caution, and richer spatial relations.

This issue was less pronounced in earlier MiniHack-style experiments because those domains were structurally simpler. There, safety often aligned more directly with avoiding obviously bad cells or states, and the action spaces did not create the same strong asymmetry between "always stop" and "make safe progress" that arises in urban navigation. In LogiCity, by contrast, the shield must reason not only about avoiding unsafe movement, but also about preserving appropriate movement.

## 4.4 Perfect and Noisy Shield Inputs

Our implementation supports both perfect and noisy symbolic inputs to the shield. In all cases, the shield operates on grounded predicate values extracted from the observation vector; what changes is how these values are converted into ProbLog fact probabilities.

In the perfect-sensor case, each grounded fact is binarized. If \(x\) is a grounded predicate value, we use
\[
\hat{x}=
\begin{cases}
1, & x>0.5,\\
0, & x\le 0.5.
\end{cases}
\]
These binary facts are then inserted directly into the ProbLog program. In this setting, uncertainty comes only from the policy probabilities in \(\Pi_s\), not from the symbolic abstraction itself.

To model uncertain symbolic perception, we support two noise mechanisms. In the **soft symmetric** mode, a binary fact \(x\in\{0,1\}\) is converted into a soft confidence value
\[
\tilde{x}=
\begin{cases}
1-\varepsilon, & x=1,\\
\varepsilon, & x=0.
\end{cases}
\]
This corresponds to a probabilistic sensor model in which true facts are softened toward uncertainty and false facts receive a small positive belief mass.

In the **bit-flip** mode, facts remain binary, but each fact is independently flipped with probability \(\varepsilon\):
\[
\tilde{x}=
\begin{cases}
1-x, & \text{with probability } \varepsilon,\\
x, & \text{with probability } 1-\varepsilon.
\end{cases}
\]
This models discrete symbolic sensing errors rather than soft probabilistic confidence.

In both cases, noise is injected after observation grounding and before ProbLog evaluation. We therefore isolate uncertainty in the shield's symbolic abstraction while keeping the underlying LogiCity dynamics unchanged.

## 4.5 Shared-Policy Multi-Agent Formulation

The thesis-level formulation of PLPG is a shared-policy multi-agent architecture in which multiple RL-controlled cars act in the same environment while sharing one set of policy parameters. Let \(K\) denote the number of RL agents. At time \(t\), each agent \(k\) receives its own local observation \(s_t^{(k)}\), and the shared policy is applied independently to each observation:
\[
\pi_\theta(\cdot\mid s_t^{(1)}),\;
\pi_\theta(\cdot\mid s_t^{(2)}),\;
\dots,\;
\pi_\theta(\cdot\mid s_t^{(K)}).
\]
This is therefore a decentralized shared-policy setting: the parameters are shared, but the observations and action decisions remain agent-specific. The single-agent case used in some experiments is simply the special case \(K=1\).

We extend the shield in exactly the same decentralized way. Each RL agent receives its own local shield evaluation from its own grounded observation. For agent \(k\), we form a local fact set \(H_{s^{(k)}}\), compute local action safety scores \(q(s^{(k)},a)\), and build a local shielded policy
\[
\pi_\theta^{+}\!\left(a\mid s_t^{(k)}\right)
=
\frac{\pi_\theta(a\mid s_t^{(k)})\, q(s_t^{(k)},a)}
{\sum_{a'} \pi_\theta(a'\mid s_t^{(k)})\, q(s_t^{(k)},a')}.
\]
Thus, our formulation is a **Decentralized PLS** design: every RL agent is shielded separately, even though all RL agents share the same neural policy parameters.

In implementation terms, the multi-agent wrapper exposes the \(K\) RL agents as a vectorized batch to a single shared PPO or PLPG learner. Rewards are still computed per agent using the same Safe Path Following reward structure, but actions are selected independently for each RL agent from its own shielded distribution. This design allows us to study whether probabilistic logic shielding continues to help when safety and progress depend not only on interaction with expert-controlled traffic, but also on coordination among multiple learning agents.
