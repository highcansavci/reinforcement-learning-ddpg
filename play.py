import gym
import numpy as np
from agent.ddpg_agent import Agent


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    env = gym.wrappers.RecordVideo(env, 'video')
    num_horizon = 20
    batch_size = 5
    n_epochs = 4
    alpha = 3e-4
    agent_ = Agent(batch_size=batch_size, actor_lr=1e-4, env=env,
                   critic_lr=1e-3, tau=1e-3, fc1_dim=400, fc2_dim=300, input_dims=env.observation_space.shape)
    agent_.load_models()
    n_games = 5

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent_.choose_action(observation)
            observation_, reward, done, info = env.step(np.squeeze(action))
            env.render()
            score += reward
            observation = observation_