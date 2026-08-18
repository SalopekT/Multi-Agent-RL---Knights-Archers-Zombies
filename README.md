# Knights Archers Zombies - Multi-Agent RL
This was a project I had done for a course in machine learning at KU Leuven. In the beginning I explored evolutionary game theory and its connection to reinforcement learning. More precisely the goal was to see how to modify some known RL algorithms in a way so that they converge to a solution concept that is better for coordination between agents. In order to do that, a small framework is made for modelling and learning matrix games. After that the goal was to train RL agents to play the game Knights Archers Zombies. I did that using a two stage architecture, where the first stage is manual feature engineering using deep learning and the second stage is reinforcement learning using the constructed features.


## 📄 Project Report

See [`report.pdf`](./report.pdf) for detailed analysis and results.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage

**Train agents:**
```bash
python3 example_training_rllib.py
```

**Test your agent:**
```bash
python3 evaluation.py -l random_agent.py -s --distortion=5
```

**Zombie detection:**
```bash
python3 evaluation.py -l random_agent.py --zombies
```


## Acknowledgments
This project was developed for purposes of the course Machine Learning project at KU Leuven.