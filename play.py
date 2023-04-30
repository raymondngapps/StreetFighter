import time
import retro
import torch
from stable_baselines3 import PPO

# print(retro.data.list_games()

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')

# Load the saved model
loaded_model = PPO.load("trained_model")

obs = env.reset()
done = False
while not done:
    action, _ = loaded_model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.005)
    
# Close the env 
env.close()
