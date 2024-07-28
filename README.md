# Flappy Bird Reinforcement Learning Project

This project uses reinforcement learning to train an agent to play Flappy Bird. The agent is trained using the Deep Q-Network (DQN) algorithm, implemented with the Stable Baselines3 library and the training environment is created using OpenAI Gym.

## Project Structure

```bash
flappybird-game/
├── assets/
│ ├── sprites/
│ │ ├── redbird-upflap.png
│ │ ├── redbird-midflap.png
│ │ ├── redbird-downflap.png
│ │ ├── bluebird-upflap.png
│ │ ├── bluebird-midflap.png
│ │ ├── bluebird-downflap.png
│ │ ├── yellowbird-upflap.png
│ │ ├── yellowbird-midflap.png
│ │ ├── yellowbird-downflap.png
│ │ ├── background-day.png
│ │ ├── background-night.png
│ │ ├── pipe-green.png
│ │ ├── pipe-red.png
│ │ ├── base.png
│ │ ├── gameover.png
│ │ ├── message.png
│ │ ├── 0.png
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── 3.png
│ │ ├── 4.png
│ │ ├── 5.png
│ │ ├── 6.png
│ │ ├── 7.png
│ │ ├── 8.png
│ │ └── 9.png
├── logs/
├── model/
├── src/
│ ├── flappybird_env.py
│ └── flappybird_rl.py
```

## Setup

### Prerequisites

- Python 3.8+
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/flappybird-game.git
    cd flappybird-game
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install --upgrade pip setuptools wheel
    pip install stable_baselines3 opencv-python mss pydirectinput gym
    pip install gym-retro --use-deprecated=legacy-resolver
    ```

## Training the Agent

To train the agent, run the following command:

```bash
python src/flappybird_rl.py --mode train
```

This will start the training process and save the trained model to the `model` directory. Training logs will be saved to the `logs` directory for visualization with TensorBoard.

## TensorBoard
To visualize training metrics, start TensorBoard:
```
tensorboard --logdir=logs/dqn_flappybird_tensorboard
```
Then open the provided URL in your browser to view the metrics.

## Testing the Agent
To test the trained agent, run the following command:

```bash
python src/flappybird_rl.py --mode test --model_path model/dqn_flappybird
```

This will load the trained model and start a testing session where the agent plays Flappy Bird.

## Project Files

- `src/flappybird_env.py`: Defines the Flappy Bird environment using OpenAI Gym.
- `src/flappybird_rl.py`: Contains the training and testing scripts for the reinforcement learning agent.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
The Flappy Bird game assets are used for educational purposes.
Thanks to the authors of Stable Baselines3 and OpenAI Gym for providing great tools for reinforcement learning research.