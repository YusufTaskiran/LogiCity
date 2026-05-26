# Results

In this chapter, we report the empirical results following the experiment-family structure introduced in Chapter 5. We begin with the single-agent benchmark, which directly addresses the two main research questions of the thesis, and then move to the method-analysis and extension families. At this stage, the chapter is written as a structured placeholder draft. The numerical values, tables, and figure references can be inserted once the final experiments have been completed.

## 6.1 Family 1: Single-Agent Main Benchmark

We first compare PPO and PLPG on the single-agent Safe Path Following task across the easy, medium, and hard difficulty levels. This family is the central benchmark of the thesis because it addresses directly whether probabilistic logic shielding improves Trajectory Success Rate and whether it reduces the number of training steps needed to reach strong performance.

Table X summarizes the main benchmark results. For each difficulty and method, we report final TSR, failure rate, timeout rate, and timesteps to convergence. Here, convergence is defined according to the stopping criterion described in the benchmark protocol. In the final version of the thesis, Table X should contain one row per method-difficulty pair.

From these results, we evaluate the two main research questions. First, we examine whether PLPG achieves higher TSR than PPO on each difficulty. Second, we compare the timesteps required by both methods to reach stable strong performance. If PLPG reaches the same or better TSR with fewer training steps, then this family provides direct evidence that probabilistic logic shielding improves sample efficiency in LogiCity.

At the qualitative level, this family also shows how the two methods fail when they do fail. PPO is expected to fail primarily through unsafe decisions or unstable convergence, whereas PLPG may trade some of those failures for more conservative behavior. The benchmark therefore does not only measure whether shielding helps, but also how it changes the balance between success, failure, and timeout.

In the final version of this section, we will summarize the main finding in one concise paragraph after Table X, for example:

> We find that PLPG achieves [higher/similar/lower] TSR than PPO on [difficulty set], while reaching convergence in [fewer/similar/more] training steps. This indicates that probabilistic logic shielding [does/does not] improve both safe task completion and sample efficiency in the single-agent LogiCity setting.

## 6.2 Family 2: Alpha Sweep

We next analyze the effect of the safety-loss weight \(\alpha\) on PLPG in the fixed easy setting. The purpose of this experiment family is to determine how strongly the explicit safety term should influence learning and whether the chosen value of \(\alpha\) produces a desirable balance between safety and progress.

Table Y or Fig. Y should summarize the results for the selected \(\alpha\) values. In the final presentation, we will report how TSR, failure rate, timeout rate, and other shield-related diagnostics vary as \(\alpha\) increases. This family is especially useful for distinguishing two effects: the effect of acting through the shielded policy, and the effect of explicitly regularizing the policy toward high shield-estimated safety.

When interpreting the results, we will focus on whether larger \(\alpha\) values reduce unsafe failures at the cost of increased conservatism, and whether smaller values leave too much unsafe behavior in the learned policy. The most desirable regime is one in which PLPG retains high TSR while avoiding both frequent failures and unnecessary timeout-dominant behavior.

In the final version of this section, we will conclude with a short statement of the form:

> The alpha sweep shows that \(\alpha = [value]\) provides the best overall trade-off between safe task completion and over-conservatism, and we therefore use this value as the main PLPG setting in the remaining experiments.

## 6.3 Family 3: Sensor Noise Robustness

In the third family, we evaluate PLPG under uncertain shield inputs. We inject noise into the grounded symbolic facts used by the shield and study how performance changes as the noise level increases. We consider both soft probabilistic noise and bit-flip noise.

The central question in this family is whether PLPG remains effective when the symbolic abstraction on which it relies is imperfect. To answer this, the final version of this section should present the results as a table or compact set of plots over noise levels, separately for the soft-noise and bit-flip settings. The most important reported quantities are TSR, failure rate, timeout rate, and the point at which performance begins to degrade substantially.

This family also plays an important interpretive role. If performance degrades under noise, we will examine whether this happens because the method becomes unsafe or because it becomes over-conservative. In our current understanding of the LogiCity setting, timeout-dominant conservatism is often the more likely failure mode of probabilistic shielding under uncertainty.

In the final version of this section, we expect to summarize the results with a statement of the form:

> PLPG remains effective up to [noise range] under [noise model], but beyond this regime the shield becomes increasingly conservative and TSR degrades mainly through [timeouts/failures].

## 6.4 Family 4: Shared-Policy Multi-Agent Results

We then extend the evaluation to the multi-agent setting, where multiple RL-controlled cars act in the same environment while sharing one decentralized policy. In this family, we compare PPO and PLPG using joint TSR, per-agent TSR, joint failure rate, and joint timeout rate.

This section should present the first empirical evidence on whether probabilistic logic shielding remains useful once several learning agents interact in the same urban environment. Unlike the single-agent case, the multi-agent setting requires the policy to handle both interaction with background traffic and interaction among RL-controlled agents themselves. This makes it a stronger test of whether the shield captures useful coordination structure rather than only individual safety constraints.

The final version of this section should present one table for the core multi-agent benchmark and, if helpful, one additional figure summarizing joint versus per-agent success. We will interpret the results primarily in terms of whether PLPG improves joint task completion without introducing excessive conservative coordination failures.

A suitable final summary sentence for this section would be:

> In the shared-policy multi-agent setting, PLPG [improves/does not improve] joint TSR relative to PPO, suggesting that decentralized probabilistic logic shielding [does/does not] transfer successfully to interacting RL agents in LogiCity.

## 6.5 Family 5: Shield Granularity Study

In the fifth family, we compare the current coarse shield against a refined action-sensitive shield. The coarse shield distinguishes only between stopping and moving, whereas the refined shield is designed to reason more explicitly about `Stop`, `Slow`, `Normal`, and `Fast`.

This family addresses a key methodological question of the thesis: whether the symbolic action granularity of the shield affects safe task completion. The motivation comes from the observation that a shield that is too coarse may suppress unsafe behavior but remain too blunt to support nuanced safe progress. A refined shield may better distinguish when the agent should move cautiously and when it can move normally or aggressively.

The final version of this section should compare the two shield variants in terms of TSR, failure rate, timeout rate, and any observable reduction in over-conservative stopping. Depending on the amount of available space, the results can be summarized in either one compact table or one table plus one illustrative figure.

We expect the conclusion of this section to take the form:

> The shield granularity study shows that the [coarse/refined] shield yields the best balance between safe progress and conservatism, indicating that symbolic action granularity [is/is not] an important design choice for probabilistic logic shielding in LogiCity.

## 6.6 Family 6: Scaling and Generalization

Finally, we evaluate the methods in larger and denser settings. This family includes larger maps, more background agents, and, where applicable, multi-agent shared-policy settings with increased interaction complexity. The purpose of this family is to test whether the benefits observed in compact benchmark scenarios persist as the task becomes more realistic and demanding.

The final results should be reported using at least one summary table, and possibly one additional figure if scaling trends are important to show visually. We will focus on whether PLPG continues to improve TSR and safety-related outcomes as map size and agent count increase, and whether the computational overhead of shielding remains acceptable in these settings.

This section is also important for the overall thesis contribution because it speaks to external validity. Even if PLPG performs well in compact settings, its practical value is much stronger if the same qualitative advantages persist in more complex urban scenarios.

In the final version, we will summarize this family with a statement such as:

> The scaling experiments show that probabilistic logic shielding [does/does not] generalize to larger and denser LogiCity scenarios, with performance trends that remain [stable/degrading] as interaction complexity increases.

## Chapter Summary

Taken together, the six experiment families provide a layered empirical picture of probabilistic logic shielding in LogiCity. The single-agent benchmark addresses the two central thesis questions directly. The alpha and noise families analyze calibration and robustness. The multi-agent family tests the decentralized shared-policy extension. The shield granularity study investigates the role of symbolic action structure, and the scaling family evaluates whether the approach generalizes beyond compact benchmark settings.

In the final version of the thesis, this chapter will be updated with the completed tables, figures, and quantitative comparisons. The chapter will then serve as the empirical basis for the broader interpretation developed in Chapter 7.
