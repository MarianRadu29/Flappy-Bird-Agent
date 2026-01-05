# Flappy Bird AI Agent (Double DQN)

This project implements an AI agent capable of playing **Flappy Bird** using **Double Deep Q-Learning (Double DQN)**. The agent learns to play by processing raw screen pixels and selecting optimal actions to maximize the score.

## Project Overview

The agent interacts with the `flappy-bird-gymnasium` environment. It receives raw screen pixels as input, processes them through a Convolutional Neural Network (CNN), and outputs the best action (Flap or Do Nothing) to maximize the score.

## Network Architecture (Double DQN)

The architecture follows a standard **Deep Q-Network** pattern with direct Q-value output.

![Network Architecture](flow-nets.png)

### 1. Input Processing
-   **Preprocessing**: Raw frames are resized to **84x84** pixels and converted to grayscale.
-   **Frame Stacking**: 4 consecutive frames are stacked to capture motion/velocity.
-   **Input Shape**: `(Batch_Size, 4, 84, 84)`.

### 2. Feature Extractor (CNN)
The image passes through convolutional layers to extract visual features (pipes, bird position):
1.  **Conv1**: 32 filters, kernel 8x8, stride 4 (ReLU).
2.  **Conv2**: 64 filters, kernel 4x4, stride 2 (ReLU).
3.  **Conv3**: 64 filters, kernel 3x3, stride 1 (ReLU).

The output is flattened and passed through fully connected layers:
*   **FC1**: Linear(3136, 512) → ReLU
*   **FC2**: Linear(512, 2) → **Q(s, a)** for each action

### 3. Output Layer
The network outputs **Q-values directly** for each action:
-   `Q(s, 0)`: Expected return for action "Do Nothing"
-   `Q(s, 1)`: Expected return for action "Flap"

```
Input (4, 84, 84)
       ↓
   Conv1 (32 filters, 8x8, stride 4) + ReLU
       ↓
   Conv2 (64 filters, 4x4, stride 2) + ReLU
       ↓
   Conv3 (64 filters, 3x3, stride 1) + ReLU
       ↓
   Flatten (3136)
       ↓
   FC1 (512) + ReLU
       ↓
   FC2 (2) → Q(s, a)
```

## Mathematical Foundation

### Q-Learning

The Q-function $Q(s, a)$ represents the expected cumulative reward when taking action $a$ in state $s$ and following the optimal policy thereafter.

**Bellman Equation:**
$$ Q(s, a) = R + \gamma \max_{a'} Q(s', a') $$

Where:
*   $R$: Immediate reward
*   $\gamma$: Discount factor
*   $s'$: Next state

### The Overestimation Problem

Standard DQN uses the same network to both **select** and **evaluate** actions:
$$ Q_{target} = R + \gamma \max_{a'} Q_{target}(s', a') $$

This leads to **overestimation** of Q-values because the max operator uses the same noisy estimates for selection and evaluation.

### Double Q-Learning Solution

**Double DQN** decouples action selection from action evaluation using two networks:

1.  **Policy Network** selects the best action:
    $$ a^* = \arg\max_{a'} Q_{policy}(s', a') $$

2.  **Target Network** evaluates that action:
    $$ Q_{target} = R + \gamma \cdot Q_{target}(s', a^*) $$

**Final Update Formula:**
$$ Q(s, a) = R + \gamma \cdot Q_{target}\left(s', \arg\max_{a'} Q_{policy}(s', a')\right) $$

This reduces overestimation by using different networks for selection and evaluation.

## Training Algorithm

The agent uses **Double Deep Q-Learning** with **Experience Replay** and **Soft Target Updates**.

### Experience Replay
Transitions (`state`, `action`, `reward`, `next_state`) are stored in a buffer and sampled randomly to break temporal correlation.

### Soft Target Update
Instead of periodically copying weights, the target network is updated smoothly at each step:
$$ \theta_{target} \leftarrow \tau \cdot \theta_{policy} + (1 - \tau) \cdot \theta_{target} $$

Where $\tau$ is the soft update coefficient.

### Loss Function
The network minimizes the **Mean Squared Error (MSE)** between predicted and target Q-values:

$$ Loss = \left( Q_{policy}(s, a) - \left( R + \gamma \cdot Q_{target}(s', a^*) \right) \right)^2 $$

### Exploration Strategy (ε-greedy)
The agent uses exponential decay for exploration:
$$ \epsilon = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \cdot e^{-steps / decay} $$

## Hyperparameters

All hyperparameters are defined in `config.py`:

- **Batch Size** - Number of transitions sampled for training
- **Learning Rate** - Optimizer step size
- **Gamma (γ)** - Discount factor for future rewards
- **Memory Size** - Replay buffer capacity
- **Tau (τ)** - Soft update coefficient for target network
- **Epsilon Start/End/Decay** - Exploration schedule parameters

## Installation

1.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment**:
    *   **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **Linux/macOS**:
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the agent from scratch:
```bash
python -B main.py
```

### Testing
To watch a trained agent play (requires `model.pth`):
```bash
python -B test.py
```

