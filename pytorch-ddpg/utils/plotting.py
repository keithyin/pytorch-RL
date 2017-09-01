import numpy as np
from collections import namedtuple
import visdom

vis = visdom.Visdom()

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "mean_rewards"])

def plot_episode_stats(stats):
    # Plot the mean of last 100 episode rewards over time.
    vis.line(X=np.arange(len(stats.mean_rewards)),
             Y=np.array(stats.mean_rewards),
             win="DDPG MEAN REWARD (100 episodes)",
             opts=dict(
                title=("DDPG MEAN REWARD (100 episodes)"),
                ylabel="MEAN REWARD (100 episodes)",
                xlabel="Episode"
                )
             )

    # Plot time steps and episode number.
    vis.line(X=np.cumsum(stats.episode_lengths),
             Y=np.arange(len(stats.episode_lengths)),
             win="DDPG Episode per time step",
             opts=dict(
                title=("DDPG Episode per time step"),
                ylabel="Episode",
                xlabel="Time Steps"
                )
             )
