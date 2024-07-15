# TFM_MUIA

## Proximal Policy Optimization (PPO) for 2D Race Circuit Environment

### Description

This repository contains the code of a TFM focused on the implementation and analysis of the PPO Deep Learning by Reinforcement algorithm using the PyTorch framework. The project consists of training a PPO agent to control a 2D car in a custom race track environment developed in Pygame. The main objective is that the agent can complete a lap of the track without collisions.


### Tabla de contenidos

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project structure](#project-structure)
5. [Hyperparameter optimisation](#hyperparameter-optimisation)
6. [Training](#training)
7. [Results](#results)
8. [Conclusions](#conclusions)

### Introduction

Proximal Policy Optimization (PPO) is a state-of-the-art reinforcement learning algorithm developed by OpenAI. It is known for its stability and efficiency in various domains. This project leverages PPO to train an agent in a high-dimensional environment, namely a 2D racetrack, to make complex decisions and navigate the track efficiently.

### Installation

To run this project, you need to have Python 3.11+ installed. Follow the steps below to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/antoniosg00/TFM_MUIA.git
   cd TFM_MUIA
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To perform hyperparameter optimisation, use the HpsOptimization.ipynb notebook. The notebook itself guides the user through comments to carry out the task.

To carry out training and validation of the agent, use the main.ipynb notebook.

### Project structure

- `Environment.py`: Contains the custom 2D racetrack environment built using Pygame. It defines the track, the car's physics, and the rules for collisions and lap completion.
- `utils.py`: Includes utility functions used throughout the project, such as data processing, logging, and visualization helpers.
- `ModelsTorch.py`: Defines the neural network architectures used for the PPO agent, implemented using PyTorch.
- `AgentTorch.py`: Implements the PPO algorithm, including the agent's decision-making processes, learning updates and validations.
- `HpsOptimization.ipynb`: A Jupyter notebook for hyperparameter optimisation. It used to experiment with different hyperparameters and evaluate their impact on the agent's performance using Weights & Biases MLOps platform.
- `main.ipynb`: The main Jupyter notebook for training and validating the PPO agent. It guides the user through the steps to set up the environment, train the agent, and evaluate its performance.
- `requirements.txt`: Lists all the Python packages required to run the project, ensuring that all dependencies are installed correctly.
- `images/`: Contains different images and templates used to create the race circuit. With *template.png* or *template.xcf*, it is possible to create new circuits using applications like Gimp or Photoshop.
