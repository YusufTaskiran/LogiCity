import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
import sys
import time
import numpy as np
import torch as th
from gymnasium import spaces
import pathlib
import io
from copy import deepcopy

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from logicity.rl_agent.policy import DreamerPolicy
from logicity.rl_agent.alg.dreamer_helper.buffer import TransitionBuffer
from logicity.rl_agent.policy.dreamer_helper.utils.module import get_parameters
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, check_for_correct_spaces
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env.patch_gym import _convert_space

SelfDreamer = TypeVar("SelfDreamer", bound="Dreamer")

class Dreamer(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "DreamerPolicy": DreamerPolicy
    }
    policy: DreamerPolicy
    def __init__(
        self,
        policy: Union[str, Type[DreamerPolicy]],
        env: Union[GymEnv, str],
        config_path: str,
        config_name: str,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        seq_len: int = 50,
        reward_weighting: bool = False,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # dynamic import of config
        config = __import__(config_path, fromlist=[config_name])
        config_class = getattr(config, config_name)
        obs_shape = env.observation_space.shape
        action_size = env.action_space.n
        obs_dtype = bool
        action_dtype = np.float32
        self.config = config_class(
            env='LogiCity',
            obs_shape=obs_shape,
            action_size=action_size,
            obs_dtype = obs_dtype,
            action_dtype = action_dtype,
            seq_len = seq_len,
            batch_size = batch_size
        )
        self.action_size = self.config.action_size
        self.pixel = self.config.pixel
        self.seq_len = self.config.seq_len
        self.batch_size = self.config.batch_size
        self.grad_clip_norm = self.config.grad_clip
        self.reward_weighting = reward_weighting
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer = TransitionBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            seq_len=self.seq_len,
            obs_type=np.float32,
            action_type=np.float32,
        )

        self.policy = self.policy_class(
            self.env,
            self.config,
            self.device,
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """ 
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        self.policy.set_training_mode(True)
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        for i in range(gradient_steps):
            obs, actions, rewards, terms = self.replay_buffer.sample(batch_size)
            obs = th.tensor(obs, dtype=th.float32).to(self.device)                         #t, t+seq_len 
            actions = th.tensor(actions, dtype=th.float32).to(self.device)                 #t-1, t+seq_len-1
            rewards = th.tensor(rewards, dtype=th.float32).to(self.device).unsqueeze(-1)   #t-1 to t+seq_len-1
            nonterms = th.tensor(1-terms, dtype=th.float32).to(self.device).unsqueeze(-1)  #t-1 to t+seq_len-1

            model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.policy.representation_loss(obs, actions, rewards, nonterms, self.reward_weighting)
            
            self.policy.model_optimizer.zero_grad()
            model_loss.backward()
            grad_norm_model = th.nn.utils.clip_grad_norm_(get_parameters(self.policy.world_list), self.grad_clip_norm)
            self.policy.model_optimizer.step()

            actor_loss, value_loss, target_info = self.policy.actorcritc_loss(posterior)

            self.policy.actor_optimizer.zero_grad()
            self.policy.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = th.nn.utils.clip_grad_norm_(get_parameters(self.policy.actor_list), self.grad_clip_norm)
            grad_norm_value = th.nn.utils.clip_grad_norm_(get_parameters(self.policy.value_list), self.grad_clip_norm)

            self.policy.actor_optimizer.step()
            self.policy.value_optimizer.step()

            with th.no_grad():
                prior_ent = th.mean(prior_dist.entropy())
                post_ent = th.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])

        self.logger.record("train/model_loss", np.mean(model_l))
        self.logger.record("train/kl_loss", np.mean(kl_l))
        self.logger.record("train/reward_loss", np.mean(reward_l))
        self.logger.record("train/obs_loss", np.mean(obs_l))
        self.logger.record("train/value_loss", np.mean(value_l))
        self.logger.record("train/actor_loss", np.mean(actor_l))
        self.logger.record("train/prior_entropy", np.mean(prior_ent_l))
        self.logger.record("train/posterior_entropy", np.mean(post_ent_l))
        self.logger.record("train/pcont_loss", np.mean(pcont_l))
        self.logger.record("train/mean_targ", np.mean(mean_targ))
        self.logger.record("train/min_targ", np.mean(min_targ))
        self.logger.record("train/max_targ", np.mean(max_targ))
        self.logger.record("train/std_targ", np.mean(std_targ))
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _on_step(self) -> None:
        """
        Get updated mean/std of the data in our replay buffer
        """
        self._n_calls += 1
        if self._n_calls % self.config.slow_target_update == 0:
            mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
            for param, target_param in zip(self.policy.ValueModel.parameters(), self.policy.TargetValueModel.parameters()):
                target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def _store_transition(
        self,
        replay_buffer: TransitionBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_



    def learn(
        self: SelfDreamer,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MBRL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDreamer:
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        rollout = self.collect_rollouts(
                self.env,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
        
        reset_result = self.env.reset()
        obs, score = (reset_result[0] if isinstance(reset_result, tuple) else reset_result), 0
        done = False
        prev_rssmstate = self.policy.RSSM._init_rssm_state(1)
        prev_action = th.zeros(1, self.action_size).to(self.device)
        episode_actor_ent = []
        
        while self.num_timesteps < total_timesteps:

            if self.num_timesteps % self.train_freq.frequency == 0:
                callback.on_rollout_end()
                self.train(self.gradient_steps, self.batch_size)     

            with th.no_grad():
                embed = self.policy.ObsEncoder(th.tensor(obs, dtype=th.float32).to(self.device))  
                _, posterior_rssm_state = self.policy.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = self.policy.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = self.policy.ActionModel(model_state)
                action = self.policy.ActionModel.add_exploration(action, self.num_timesteps).detach()
                action_ent = th.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            # Step the env, using the discrete numbers, not the one-hot
            env_action = th.argmax(action, dim=-1).cpu().numpy()
            step_result = self.env.step(env_action)
            if len(step_result) == 5:
                next_obs, rew, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, rew, done, info = step_result
            score += rew

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(info, done)

            if done:
                self._episode_num += 1
                self.replay_buffer.add(obs, env_action, rew, done)
                self.logger.record("train/train_rewards", np.mean(score))
                self.logger.record("train/action_ent", np.mean(np.mean(episode_actor_ent)))
                
                reset_result = self.env.reset()
                obs, score = (reset_result[0] if isinstance(reset_result, tuple) else reset_result), 0
                done = False
                prev_rssmstate = self.policy.RSSM._init_rssm_state(1)
                prev_action = th.zeros(1, self.action_size).to(self.device)
                episode_actor_ent = []
            else:
                self.replay_buffer.add(obs, env_action, rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

            callback.update_locals(locals())
            self._on_step()
            self.num_timesteps += self.env.num_envs
            if callback.on_step() is False:
                break
            
            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        callback.on_training_end()

        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # use on-policy data here
        action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def collect_rollouts(
            self,
            env,
            callback,
            replay_buffer: TransitionBuffer,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
        ):
            """
            Collect experiences and store them into a ``ReplayBuffer``.

            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param train_freq: How much experience to collect
                by doing rollouts of current policy.
                Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
                or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
                with ``<n>`` being an integer greater than 0.
            :param action_noise: Action noise that will be used for exploration
                Required for deterministic policy (e.g. TD3). This can also be used
                in addition to the stochastic policy for SAC.
            :param learning_starts: Number of steps before learning for the warm-up phase.
            :param replay_buffer:
            :param log_interval: Log data every ``log_interval`` episodes
            :return:
            """
            # Switch to eval mode (this affects batch norm / dropout)
            self.policy.set_training_mode(False)

            num_collected_steps, num_collected_episodes = 0, 0
            assert self.n_envs == 1, "Only one env is supported for now"
            if self.use_sde:
                self.actor.reset_noise(env.num_envs)

            callback.on_rollout_start()
            continue_training = True

            while num_collected_steps < learning_starts:
                # Select action randomly or according to policy
                unscaled_action = np.array([self.action_space.sample() for _ in range(self.n_envs)])
                buffer_actions = unscaled_action
                actions = buffer_actions
                # Rescale and perform action
                new_obs, rewards, dones, infos = env.step(actions)

                self.num_timesteps += env.num_envs
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if not callback.on_step():
                    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, dones)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                for idx, done in enumerate(dones):
                    if done:
                        # Update stats
                        num_collected_episodes += 1
                        self._episode_num += 1
                        # Log training infos
                        if log_interval is not None and self._episode_num % log_interval == 0:
                            self._dump_logs()
            callback.on_rollout_end()

            return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfDreamer],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfDreamer:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """

        data, params, pyth_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            data["policy_kwargs"].update(kwargs["policy_kwargs"])

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
            **kwargs,
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pyth variables back in place
        if pyth_variables is not None:
            for name in pyth_variables:
                # Skip if Pyth variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional Pyth variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pyth_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pyth_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model
