# Results

In this chapter, we report the empirical results using the research-question structure introduced in Chapter 5. This makes the chapter directly answer the thesis questions rather than merely walking through experiment families. At this stage, the chapter remains a structured placeholder draft: the final numerical values, tables, and figure references can be inserted once the experiments are complete.

## 6.1 RQ1: Does probabilistic logic shielding improve safe task completion relative to PPO?

We answer the first research question by comparing PPO and PLPG on the Safe Path Following task under the same shared-policy formulation. We begin with the controlled \(K=1\) benchmark across the easy, medium, and hard difficulty levels, and then relate those results to the multi-agent benchmark introduced in RQ2.

The main quantities reported for this question are TSR, failure rate, timeout rate, mean reward, mean episode length, and timesteps required to reach stable strong performance. In the final version of the thesis, Table X should summarize these values for PPO and PLPG across all three difficulties in the \(K=1\) setting.

This question addresses the most basic empirical claim of the thesis: whether probabilistic logic shielding improves safe task completion relative to a purely neural baseline. When interpreting the results, we will not only compare final TSR, but also examine how the two methods fail. PPO may fail more often through unsafe decisions or unstable learning, whereas PLPG may reduce unsafe failures while risking increased conservatism. The balance between success, failure, and timeout is therefore central to the interpretation.

In the final version of this section, we will close with a concise statement of the form:

> In the controlled \(K=1\) setting, PLPG achieves [higher/similar/lower] TSR than PPO and reaches strong performance in [fewer/similar/more] training steps, indicating that probabilistic logic shielding [does/does not] improve safe task completion and sample efficiency in LogiCity.

## 6.2 RQ2: Does probabilistic logic shielding remain beneficial as the number of RL-controlled agents increases?

The second research question is the core multi-agent question of the thesis. Here, we evaluate PPO and PLPG in the shared-policy multi-agent setting with \(K>1\), beginning with \(K=2\) RL-controlled cars and extending to \(K=3\) if computational budget permits. We also compare results across \(K=1 \rightarrow K=2 \rightarrow K=3\) to make the scaling behavior explicit.

For this question, the primary metric is **joint TSR**, i.e., the fraction of episodes in which all RL-controlled agents reach their goals before any failure and before the horizon is exceeded. We also report per-agent TSR, joint failure rate, joint timeout rate, mean reward, and mean episode length. Table Y should summarize the core multi-agent benchmark, and one compact figure can be used to show how performance changes as \(K\) increases.

This section provides the main empirical evidence for whether decentralized probabilistic logic shielding transfers successfully from the controlled \(K=1\) case to the true thesis-level setting in which several RL agents interact in the same environment. The interpretation should focus on whether PLPG improves **joint** task completion rather than only local per-agent behavior, and on whether any benefit remains stable or degrades as coordination demands increase.

A suitable final summary sentence for this section would be:

> In the shared-policy multi-agent setting, PLPG [improves/does not improve] joint TSR relative to PPO across \(K=[...]\), suggesting that decentralized probabilistic logic shielding [does/does not] remain effective as the number of interacting RL agents increases.

## 6.3 RQ3: Does shielding remain useful when the symbolic inputs are mildly noisy?

The third research question examines whether the multi-agent shield remains useful when the symbolic inputs are not perfect. We compare the shared-policy multi-agent setting under two conditions: perfect symbolic inputs and one fixed mildly noisy condition, such as soft symmetric noise with \(\varepsilon = 0.1\).

The final version of this section should present the results in one compact table comparing PPO and PLPG under the perfect and noisy shield-input conditions. The most important reported quantities are joint TSR, joint failure rate, joint timeout rate, and any change in the qualitative balance between unsafe behavior and over-conservative stopping.

This section plays an interpretive role rather than a calibration role. We are not trying to establish a full robustness curve. Instead, we are asking whether PLPG still offers a meaningful advantage once the shield must reason from mildly imperfect grounded symbolic facts in a genuinely multi-agent environment.

In the final version of this section, we expect to conclude with a statement of the form:

> Under one fixed noisy-symbolic condition, PLPG [retains/loses] its advantage over PPO, with performance degrading mainly through [timeouts/failures], which suggests that the shield is [robust/not robust] to mild symbolic uncertainty in multi-agent LogiCity.

## 6.4 RQ4: Does shield granularity matter for safe progress?

The fourth research question addresses one key design decision of the thesis: whether a coarse stop-versus-move shield is sufficient, or whether a more refined action-sensitive shield leads to better safe task completion. We therefore compare the current coarse shield against a refined shield that distinguishes more explicitly among `Stop`, `Slow`, `Normal`, and `Fast`.

The final version of this section should compare the two shield variants in terms of TSR, failure rate, timeout rate, and any visible reduction in over-conservative stopping. Depending on space, the results can be summarized in either one compact table or one table plus one illustrative figure.

The interpretation here is especially important because LogiCity is not a pure avoidance domain. A shield that is too coarse may prevent unsafe actions while still failing the real task through excessive stopping. This section therefore asks whether symbolic action granularity is an important design choice for balancing safety and progress.

We expect the conclusion of this section to take the form:

> The shield granularity study shows that the [coarse/refined] shield yields the best balance between safe progress and conservatism, indicating that symbolic action granularity [is/is not] an important design choice for probabilistic logic shielding in LogiCity.

## Chapter Summary

Taken together, the results chapter answers four thesis questions. RQ1 establishes whether PLPG improves safe task completion relative to PPO in a controlled instantiation of the shared-policy formulation. RQ2 evaluates whether that benefit remains present in the full multi-agent setting and how it changes as \(K\) increases. RQ3 tests whether the benefit survives one fixed mild noisy-symbolic condition. RQ4 studies whether shield granularity matters for achieving safe progress rather than only preventing unsafe motion.

In the final version of the thesis, this chapter will be updated with completed tables, figures, and quantitative comparisons. The chapter will then provide the empirical basis for the broader interpretation developed in Chapter 7.
