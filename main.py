import gym
import numpy as np
from agent.ddpg_agent import Agent
from utils.utils import plot_learning_curve


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    num_horizon = 10000
    batch_size = 1000
    n_epochs = 4
    alpha = 3e-4
    agent_ = Agent(batch_size=batch_size, actor_lr=1e-4,
                   critic_lr=1e-3, tau=1e-3, fc1_dim=400, fc2_dim=300, input_dims=[env.observation_space.shape[0]])

    #agent_.load_models()
    n_games = 10000
    figure_file = "plots/lunarlander.png"

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent_.choose_action(observation)
            observation_, reward, done, info = env.step(np.squeeze(action))
            n_steps += 1
            score += reward
            agent_.remember(observation, action, reward, observation_, done)
            if n_steps % num_horizon == 0:
                agent_.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent_.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
