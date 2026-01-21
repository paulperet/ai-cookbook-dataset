# Q-Learning Tutorial: Implementing Reinforcement Learning Without Knowing the MDP

In this guide, you will implement Q-Learning, a foundational reinforcement learning algorithm that enables an agent to learn optimal behavior through interaction with an environment, without requiring prior knowledge of the environment's dynamics (the Markov Decision Process or MDP). We will use the FrozenLake environment from OpenAI Gym as our testbed.

## Prerequisites

First, ensure you have the necessary libraries installed. We'll use `numpy` for numerical operations, `random` for stochasticity, and the `d2l` library for environment setup and visualization.

```bash
pip install numpy gym
# The d2l library is assumed to be available. If not, install from the Dive into Deep Learning resources.
```

Now, let's import the required modules and set up our environment.

```python
import numpy as np
import random
from d2l import torch as d2l

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

# Define hyperparameters
gamma = 0.95    # Discount factor for future rewards
num_iters = 256 # Number of training iterations
alpha = 0.9     # Learning rate
epsilon = 0.9   # Exploration rate for epsilon-greedy policy

# Create the FrozenLake environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

## Understanding the FrozenLake Environment

The agent operates on a 4x4 grid. Each cell is either:
*   **F (Frozen):** Safe to walk on.
*   **H (Hole):** Falls through, ending the episode.
*   **G (Goal):** Reaching here gives a reward of +1 and ends the episode.
*   **S (Start):** The starting position.

The agent has four possible actions: Up (0), Down (1), Left (2), Right (3). The state is the current grid cell. The transition is deterministic (moving in the chosen direction), but the agent does not know the map layout. The objective is to learn a policy to navigate from Start (S) to Goal (G) while avoiding holes (H).

## Step 1: Implementing the Epsilon-Greedy Exploration Policy

A core challenge in reinforcement learning is the **exploration-exploitation trade-off**. The agent must balance trying new actions (exploration) with choosing actions known to be good (exploitation). We implement an **ε-greedy policy**:

*   With probability `ε`, take a random action (exploration).
*   With probability `1-ε`, take the action with the highest current Q-value (exploitation).

```python
def e_greedy(env, Q, s, epsilon):
    """
    Selects an action using an epsilon-greedy policy.

    Args:
        env: The environment object.
        Q: The current Q-value table (states x actions).
        s: The current state.
        epsilon: The exploration probability.

    Returns:
        The selected action.
    """
    if random.random() < epsilon:
        # Explore: choose a random action
        return env.action_space.sample()
    else:
        # Exploit: choose the action with the maximum Q-value for state s
        return np.argmax(Q[s, :])
```

## Step 2: The Q-Learning Algorithm

Q-Learning directly learns the optimal action-value function, `Q*(s,a)`, which represents the expected total reward from taking action `a` in state `s` and following the optimal policy thereafter.

The update rule, derived from temporal difference learning, is:

`Q(s,a) ← Q(s,a) + α * [ r + γ * max_a' Q(s', a') - Q(s,a) ]`

Where:
*   `α` is the learning rate.
*   `r` is the immediate reward.
*   `γ` is the discount factor.
*   `s'` is the next state.

This update moves the current Q-value estimate towards the "target" value: the immediate reward plus the discounted value of the best future action.

```python
def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    """
    Executes the Q-Learning algorithm.

    Args:
        env_info: Dictionary containing environment details.
        gamma: Discount factor.
        num_iters: Number of training episodes/iterations.
        alpha: Learning rate.
        epsilon: Exploration rate.
    """
    env_desc = env_info['desc']  # Grid layout for visualization
    env = env_info['env']        # The main environment object
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    # Initialize Q-table with zeros
    Q = np.zeros((num_states, num_actions))

    # Arrays to track the value function and policy over time (for visualization)
    V = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    # Main training loop
    for k in range(1, num_iters + 1):
        # Reset the environment at the start of each episode
        state, done = env.reset(), False

        # Run a single episode until termination (falling in a hole or reaching the goal)
        while not done:
            # 1. Select action using epsilon-greedy policy
            action = e_greedy(env, Q, state, epsilon)

            # 2. Execute action, observe reward and next state
            next_state, reward, done, _ = env.step(action)

            # 3. Calculate the target value for the Q-update
            # If next_state is terminal, there is no future value.
            if done and reward == 0:  # Fell into a hole
                target = reward
            else:  # Non-terminal state or reached the goal
                target = reward + gamma * np.max(Q[next_state, :])

            # 4. Update the Q-value for the (state, action) pair
            Q[state, action] = Q[state, action] + alpha * (target - Q[state, action])

            # 5. Transition to the next state
            state = next_state

        # After each episode, record the current value function and policy for all states
        for s in range(num_states):
            V[k, s] = np.max(Q[s, :])       # Value is max Q-value for that state
            pi[k, s] = np.argmax(Q[s, :])   # Policy is the action with max Q-value

    # Visualize the learning progress
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])
```

## Step 3: Run Q-Learning

Now, execute the algorithm with our defined hyperparameters.

```python
q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)
```

When you run this code, you will see a visualization showing how the estimated value function (`V`) and the derived policy (`pi`) evolve over the 256 training iterations. The agent starts with no knowledge (all Q-values are zero) and gradually learns which states and actions lead to the goal.

**Key Observation:** Q-Learning successfully finds a path from the start to the goal, but it requires more iterations (around 250) compared to model-based algorithms like Value Iteration. This is because Q-Learning learns purely from experience without a model of the environment's transitions.

## How Q-Learning Works: The Self-Correcting Property

The algorithm's power comes from its interactive nature:
1.  The ε-greedy policy ensures the agent explores the state-action space.
2.  If an action is overvalued (has a high Q-value but is actually poor), taking it will lead to low rewards in subsequent states.
3.  The next Q-update will then *reduce* the value of that overestimated action.
4.  Conversely, good actions are reinforced through repeated updates.
This feedback loop allows Q-Learning to converge to the optimal policy even when starting with random Q-values.

## Summary

In this tutorial, you implemented Q-Learning, a model-free reinforcement learning algorithm. You learned to:
*   Set up a simple grid-world environment (FrozenLake).
*   Implement an ε-greedy exploration strategy to balance exploration and exploitation.
*   Apply the core Q-Learning update rule to iteratively improve an action-value function (Q-table).
*   Understand how the algorithm corrects its own estimates through interaction.

Q-Learning is a cornerstone of modern RL, forming the basis for more advanced algorithms like Deep Q-Networks (DQNs) that can tackle complex problems like playing video games directly from pixels.

## Exercises

To deepen your understanding, try modifying the code and observe the effects:

1.  **Increase Environment Complexity:** Change the grid size to 8x8 in the environment creation (`'FrozenLake8x8-v1'`). How many iterations does the agent now need to find a reasonable policy compared to the 4x4 grid?
2.  **Vary the Discount Factor (`gamma`):** Run the algorithm with `gamma = 0`, `0.5`, and `1`.
    *   `gamma=0`: The agent is "myopic," caring only about immediate reward. Does it still find the goal?
    *   `gamma=1`: The agent values future rewards equally to immediate ones. How does this affect the learned value function?
3.  **Adjust Exploration (`epsilon`):** Run the algorithm with `epsilon = 0`, `0.5`, and `1`.
    *   `epsilon=0`: The agent never explores (purely greedy). Will it get stuck in a suboptimal policy?
    *   `epsilon=1`: The agent always explores (purely random). Can it learn an optimal policy?