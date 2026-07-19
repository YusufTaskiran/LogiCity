
def collect_trajectories(env, model, n_steps):
    # Initialize state, action, reward, and done lists
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    # Reset the environments
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    for _ in range(n_steps):
        # Use the model to determine the action
        action, _ = model.predict(obs, deterministic=True)
        
        # Take a step in the environment
        step_result = env.step(action)
        if len(step_result) == 5:
            new_obs, reward, terminated, truncated, infos = step_result
            done = terminated or truncated
        else:
            new_obs, reward, done, infos = step_result

        # Store state, action, reward, next_state, and done signal
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(new_obs)
        dones.append(done)

        # Update state
        obs = new_obs

    return states, actions, next_states, rewards, dones
