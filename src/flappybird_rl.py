import os
import argparse
import gymnasium as gym
import numpy as np
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from flappybird_env import FlappyBirdEnv
import pygame

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Custom callback to log additional metrics
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if 'dones' in self.locals and self.locals["dones"][0]:
            self.logger.record("reward", self.locals["rewards"])
            if "episode" in self.locals["infos"][0]:
                self.logger.record("episode_length", self.locals["infos"][0]["episode"]["l"])
        return True

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# Function to train the model
def train_model():
    logger.info("Creating training environment...")
    env = FlappyBirdEnv(render_mode=False)

    logger.info("Checking environment...")
    check_env(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    env = VecTransposeImage(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    project_root = os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(project_root, "..", "logs", "dqn_flappybird_tensorboard")
    model_dir = os.path.join(project_root, "..", "model")

    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Model directory: {model_dir}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info("Directories created or already exist.")

    logger.info("Defining the model...")
    model = DQN(
        'CnnPolicy', 
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        tensorboard_log=log_dir,
    )

    # model.set_parameters("model/FlappyBird_DQN_Easy")
    logger.info("Starting training...")
    model.learn(total_timesteps=1000000, callback=TensorboardCallback())
    logger.info("Training completed.")

    model_path = os.path.join(model_dir, "dqn_flappybird")
    logger.info(f"Saving model to: {model_path}")
    model.save(model_path)
    logger.info(f"Model saved at {model_path}")

    return model_path

# Function to test the trained model
def test_model(model_path):
    logger.info(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    logger.info("Creating testing environment...")
    test_env = FlappyBirdEnv(render_mode=True)  
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecFrameStack(test_env, 4, channels_order='last')  # Match the frame stack used during training

    obs = test_env.reset()
    done = False
    while test_env.envs[0].running:  # Run until the user closes the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Window closed by user.")
                test_env.envs[0].close()
                return

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()

        if done:
            logger.info("Model lost the game, resetting environment...")
            obs = test_env.reset()
            done = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the Flappy Bird DQN model.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Specify whether to train or test the model.")
    parser.add_argument('--model_path', type=str, help="Path to the model for testing. Required if mode is 'test'.")
    args = parser.parse_args()

    if args.mode == 'train':
        try:
            model_path = train_model()
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
    elif args.mode == 'test':
        if not args.model_path:
            logger.error("Model path must be provided for testing.")
        else:
            try:
                test_model(args.model_path)
            except Exception as e:
                logger.error(f"An error occurred during testing: {e}")
