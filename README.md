# Multiple Task Replay Buffer

This project builds upon stable_baselines3 buffers to propose buffers designed for multi-task concurrent training and Meta Reinforcement Learning (RL).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Overview

In reinforcement learning, replay buffers are used to store and sample experience tuples, facilitating stable and efficient training. This project introduces two specialized replay buffers tailored for multi-task learning scenarios:

- **MultipleReplayBuffer:** Stores training information for multiple tasks.
- **HistoryMultipleReplayBuffer:** Stores training information and history for multiple tasks, suitable for models that require past tuples as context.

## Features

- **Task-Specific Storage:** Each buffer can handle data from multiple tasks, maintaining separation to ensure task-specific learning.
- **Historical Context:** The `HistoryMultipleReplayBuffer` provides historical data, enabling models to utilize past experiences as context for current decision-making.

## Usage

To integrate the replay buffers into your reinforcement learning workflow:

1. **Import the desired buffer class:**

   ```python
   from buffer import MultipleReplayBuffer, HistoryMultipleReplayBuffer
   ```

2. **Initialize the buffer:**

   ```python
   buffer = MultipleReplayBuffer(buffer_size=10000, num_tasks=5)
   ```

   Replace `MultipleReplayBuffer` with `HistoryMultipleReplayBuffer` if historical context is required.

3. **Add experiences to the buffer:**

   ```python
   buffer.add(task_id, state, action, reward, next_state, done)
   ```

4. **Sample experiences for training:**

   ```python
   experiences = buffer.sample(batch_size=64, task_id=task_id)
   ```

   Ensure to handle the sampled experiences appropriately in your training loop.
