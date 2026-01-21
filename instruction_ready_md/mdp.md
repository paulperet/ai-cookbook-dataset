# Markov Decision Processes (MDPs): A Foundational Guide

In this guide, we will explore how reinforcement learning problems are formulated using Markov Decision Processes (MDPs). We'll break down the core components of an MDP and explain how they work together to model decision-making under uncertainty.

## What is a Markov Decision Process?

A Markov Decision Process (MDP) is a mathematical framework used to model sequential decision-making problems. It describes an environment where an agent interacts with a system, makes decisions (actions), and receives feedback (rewards). The "Markov" property implies that the future state depends only on the current state and action, not on the entire history.

### Core Components of an MDP

An MDP is formally defined by the tuple `(S, A, T, r)`.

#### 1. State Space (`S`)
*   **Definition:** The set of all possible situations or configurations the agent can be in.
*   **Example:** In a robot navigation task, the state could be the robot's current (x, y) coordinates on a grid.

#### 2. Action Space (`A`)
*   **Definition:** The set of all possible moves or decisions the agent can make from any given state.
*   **Example:** For our grid robot, actions might be `{up, down, left, right, stay}`.

#### 3. Transition Function (`T`)
*   **Definition:** Also called the transition probability, `T(s, a, s')` defines the probability of moving to state `s'` after taking action `a` in state `s`. It models the uncertainty in the environment.
*   **Mathematically:** `T(s, a, s') = P(s' | s, a)`
*   **Key Property:** It is a probability distribution. For a fixed state `s` and action `a`, the sum of probabilities to all possible next states `s'` must equal 1:
    `∑_{s' ∈ S} T(s, a, s') = 1`

#### 4. Reward Function (`r`)
*   **Definition:** The immediate feedback signal `r(s, a)` received after taking action `a` in state `s`. It quantifies the short-term desirability of that action-state pair.
*   **Design Note:** The reward function is crafted by the algorithm designer to guide the agent toward the overall goal (e.g., `+10` for reaching the goal, `-1` for each movement step to encourage efficiency).

## Understanding Returns and Discounting

When an agent acts, it generates a **trajectory** (τ):
`τ = (s₀, a₀, r₀, s₁, a₁, r₁, s₂, a₂, r₂, ...)`

The **Return** `R(τ)` is the total reward accumulated along this trajectory.

### The Problem with Infinite Horizons
For ongoing tasks, trajectories can be infinitely long, leading to an infinite sum of rewards. This makes comparisons between trajectories impossible.

### Solution: The Discount Factor (γ)
We introduce a discount factor `0 ≤ γ < 1` to compute a **Discounted Return**:
`R(τ) = r₀ + γ·r₁ + γ²·r₂ + ... = Σ_{t=0}^{∞} γᵗ·rₜ`

*   **Why it works:** Future rewards are weighted less heavily than immediate rewards. The further in the future, the more the reward is "discounted."
*   **Interpretation:**
    *   A **small γ** (e.g., 0.1) makes the agent short-sighted, focusing on immediate high rewards.
    *   A **large γ** (e.g., 0.99) makes the agent far-sighted, encouraging it to invest in actions that lead to higher rewards in the distant future.

## The Markov Assumption: A Closer Look

The core assumption in an MDP is that the **next state depends only on the current state and current action**, not on the entire history of states and actions.

### Is This Assumption Restrictive?
At first glance, it seems many real-world problems don't fit this mold. Consider a robot where the state is its *position*, and the action is *acceleration*. The next position depends not just on the current position and acceleration, but also on the current *velocity* (which is a function of past positions).

### The Power of State Definition
The Markov property can often be satisfied by **redefining the state** to include all necessary information from the past. For the accelerating robot, if we define the state as the tuple `(position, velocity)`, then the system becomes Markovian. The next `(position, velocity)` depends only on the current `(position, velocity)` and the current acceleration.

This demonstrates that MDPs are a highly flexible model capable of representing a vast array of sequential decision problems through careful state representation.

## Summary

*   A **Markov Decision Process (MDP)** is the standard model for reinforcement learning problems, defined by the tuple `(S, A, T, r)`.
*   The **State Space (S)** and **Action Space (A)** define the environment and the agent's possible choices.
*   The **Transition Function (T)** models the environment's dynamics and uncertainty.
*   The **Reward Function (r)** provides the signal for learning, defining what is "good" or "bad."
*   The **Discounted Return** is used to evaluate finite or infinite sequences of actions, with the discount factor `γ` controlling the agent's time horizon.
*   The **Markov Assumption** is not overly restrictive; many non-Markovian systems can be reformulated as Markovian by incorporating relevant history into the state definition.

## Exercises

1.  **MountainCar MDP:** Consider the classic [MountainCar](https://www.gymlibrary.dev/environments/classic_control/mountain_car/) control problem.
    *   What would constitute the set of states `S`?
    *   What would be the set of actions `A`?
    *   Propose one or two possible reward functions `r(s, a)`.

2.  **Atari Pong MDP:** How would you design an MDP for the [Pong](https://www.gymlibrary.dev/environments/atari/pong/) game?
    *   Define the core components `(S, A, T, r)` for this environment. What are the challenges in specifying `T`?