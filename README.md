# Flappy Bird AI Agent (Double DQN)

This project implements an AI agent capable of playing **Flappy Bird** using **Double Deep Q-Learning (Double DQN)**. The agent learns to play directly from **raw visual input**, using a Deep Q-Network to process frames and select optimal actions.

## Project Overview

The core objective is to bridge the gap between computer vision and reinforcement learning. The agent "sees" the game through a series of processed frames and learns through trial and error which actions (flap or stay) lead to higher scores.

### Key Features
-   **Visual Learning**: Processes raw pixels instead of coordinate-based state vectors.
-   **Double DQN**: Implements a robust learning algorithm that reduces overestimation bias.
-   **Advanced Preprocessing**: Uses computer vision techniques (HSV masking) to isolate relevant game objects.
-   **Temporal Awareness**: Employs frame stacking to help the agent understand motion and velocity.
-   **Stable Updates**: Uses soft target updates (Polyak averaging) for smoother convergence.

## State Representation & Preprocessing

One of the project's highlights is its custom `Wrapper` that transforms the standard game output into an optimized input for the neural network.

### 1. Object Isolation (HSV Masking)
To help the agent focus on critical elements, the environment frames undergo an HSV-based filtering process:
-   **Background Removal**: Masks out the sky, clouds, and static decorative background elements.
-   **Object Extraction**: Isolates the bird, the pipes, and the ground.
-   **Result**: A clean, binary-like image where only the "threats" and the agent are visible, significantly reducing the learning complexity.

### 2. Spatial & Temporal Processing
-   **Cropping**: Removes the bottom 410 pixels (ground/UI) to focus on the flight area.
-   **Resizing**: Scales the image down to **84x84** pixels to optimize processing speed.
-   **Frame Stacking**: Stacks **4 consecutive frames** into a single input tensor of shape `(4, 84, 84)`. This allows the CNN to "see" the bird's vertical velocity and gravity's effect over time.

## Network Architecture (Double DQN)

The agent uses a Deep Q-Network consisting of a Convolutional Neural Network (CNN) feature extractor followed by fully connected layers that output Q-values for each action.

```text
       Input (4, 84, 84)
              ↓
   Conv1 (32 filters, 8x8, stride 4)
              ↓
            ReLU
              ↓
   Conv2 (64 filters, 4x4, stride 2)
              ↓
            ReLU
              ↓
   Conv3 (64 filters, 3x3, stride 1)
              ↓
            ReLU
              ↓
           Flatten (3136)
              ↓
       FC1 (512 neurons)
              ↓
            ReLU
              ↓
       FC2 (2 neurons)
              ↓
        Q(s, "Nothing")
        Q(s, "Flap")
```

### 1. Feature Extractor (CNN)
The visual pipeline consists of three convolutional layers to extract critical features like pipe positions and bird orientation:
1.  **Conv1**: 32 filters, 8x8 kernel, stride 4.
2.  **Conv2**: 64 filters, 4x4 kernel, stride 2.
3.  **Conv3**: 64 filters, 3x3 kernel, stride 1.

### 2. Decision Head
The extracted features are flattened and passed through fully connected layers:
-   **FC1**: 512 neurons with ReLU activation.
-   **FC2**: 2 output neurons representing the **Q-values** for each possible action.

## Mathematical Foundation

### Double Q-Learning Solution

Standard DQN often suffers from **overestimation** of Q-values because it uses the same network to both select and evaluate actions. **Double DQN** solves this by decoupling the two processes:

1.  **Action Selection**: The **Policy Network** (θ) determines the best action for the next state:
    ```text
    a* = arg max Q(s', a'; θ)
    ```

2.  **Action Evaluation**: The **Target Network** (θ⁻) calculates the value of that specific action:
    ```text
    Q_target = R + γ · Q(s', a*; θ⁻)
    ```

This prevents the agent from being overly optimistic about noisy reward estimates, leading to more stable and reliable training.

### Soft Target Update
Instead of hard-copying weights every few thousand steps, we use **Polyak Averaging**:
```text
θ⁻ ← τθ + (1 - τ)θ⁻
```
With τ = 0.005, the target network follows the policy network slowly, providing a stable target for the loss function.

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
    *   **Windows**: `.\venv\Scripts\activate`
    *   **Linux/macOS**: `source venv/bin/activate`

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
Start the training process:
```bash
python -B main.py
```
The agent will save checkpoints every 500 episodes and keep the best model in `model.pth`.

### Testing
Evaluate a trained agent:
```bash
python -B test.py
```
