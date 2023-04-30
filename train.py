import retro
import torch
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0, envs=None, render_freq: int = 30):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.writer = SummaryWriter()
        self.envs=envs
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.model.get_env().render()
        return True

    def _on_rollout_end(self) -> None:
        episode_reward = np.sum(self.model.rollout_buffer.rewards)
        episode_length = self.model.rollout_buffer.buffer_size
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.writer.add_scalar('Reward/Episode', episode_reward, self.num_timesteps)
        self.writer.add_scalar('Length/Episode', episode_length, self.num_timesteps)

    def _on_training_end(self) -> None:
        self.writer.close()

def train_model(model, callback, save_interval=5000):
    total_timesteps = 70000
    num_checkpoints = int(total_timesteps / save_interval)

    for i in range(num_checkpoints):
        model.learn(total_timesteps=save_interval, callback=callback)
        model.save(f'sf2_model_{i * save_interval}')

def make_env():
    env = retro.RetroEnv(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile', scenario='scenario')
    env.seed(random.sample(range(100), 10))
    return env

def make_vec_env():
    return SubprocVecEnv([make_env] * 4)

def make_model(env):
    model = PPO('CnnPolicy', env, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    return model

if __name__ == '__main__':
    envs = make_vec_env()

    model = make_model(envs)
    callback = CustomCallback(envs=envs)
    
    train_model(model, callback, save_interval=2000)

    # After training, save the final model
    model.save("trained_model")
    
    envs.close()
