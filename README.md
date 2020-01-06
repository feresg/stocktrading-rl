## Stock Trading using Reinforcement Learning

Playing with actor critic deep reinforcement learning models for automating and optimizing stock trading strategies to maximize profit in a custom OpenAI gym.
We will use pretrained models using the stable_baselines library (A2C, PPO2, TRPO) and a custom DDPG model in Keras (buggy)


### Project Structure
---
*   **Stock Trading with RL.ipynb**: Jupyter Notebook for interacting with the different components
*   **env.py**: StockTradingEnv OpenAI gym environment, where we define the observation space, agent actions (BUY, SELL, HOLD and percentage of shares (continuous action space)).
*   **graph.py**: Used to render live trades from the agent
*   **agent.py**: Implementation of a DDPG (Deep Deterministic Policy Gradient) RL agent.
*   **models.py**: Contains the Actor and Critic models used by the DDPG agent (actor maps states to actions, critic returns Q value of the state action mapping)
*   **utils.py**: Utility functions used by the project


