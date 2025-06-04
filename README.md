# Lunar-Lander

A Python project for simulating and solving the Lunar Lander problem using evolutionary algorithms.

## Overview

This repository contains Python code to train and evaluate an agent for the Lunar Lander environment using genetic algorithms and policy networks. The agent's policy is optimized to maximize the reward in the LunarLander-v2 OpenAI Gym environment.

- Repository: [dhruv-2615p/Lunar-Lander](https://github.com/dhruv-2615p/Lunar-Lander)
- Language: Python
- Visibility: Public

## Features

- **Policy Network:** Uses a simple neural network (linear layer) to map state observations to actions.
- **Genetic Algorithm:** Optimizes the agent's policy parameters through:
  - Simulated Binary Crossover (SBX) for recombination between elite parents.
  - Polynomial Mutation for diversity and exploration.
  - Tournament Selection to choose parents for crossover.
  - Elitism to retain the best-performing individuals.
  - Adaptive mutation rate to escape stagnation.
- **Parallel Evaluation:** Uses Python's `multiprocessing.Pool` to evaluate policy fitness across multiple CPUs for faster training.
- **Evaluation Scripts:** Includes code to evaluate saved policies over multiple episodes with rendering.
- **Modular Design:** Policies and evaluation logic are separated for flexibility and easy experimentation.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dhruv-2615p/Lunar-Lander.git
   cd Lunar-Lander

2. **Train a new policy**
   ```bash
    python train_model.py --train --filename best_policy.npy

3. **Play/evaluate a trained policy**
  ```bash
    python train_model.py --play --filename best_policy.npy
  ```

4. **Evaluate using a custom module**  
  ```bash
  python evaluate_agent.py --filename best_policy.npy --policy_module my_policy
  ```

## Contributing
- Contributions are welcome!

## Repository Owner
- dhruv-2615p
