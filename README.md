This repository contains the Python version of a Reinforcement Learning (RL) project that is under development.

The goal of the overall project is to have some RL agent(s) learn to play a game in a cellular automata (CA) environment. 

This project, and the RL training involved, is meant to engage users, be visual, and be part of a broader distributed machine learning project and website. [That website](https://www.nets-vs-automata.net) uses client-side computation with TensorFlow.js. Tensorflow.js will be used in the final implementation of this project, but in the current phase, the exploratory phase of just trying to nail down the game details and get RL to work, Python will be the language used. This repo is the source.

The web page currently on hold:
https://www.nets-vs-automata.net/rl.html

#
"Pattern" Game 🪄

Goal: 
Draw/type/generate a state/pattern on the board (starting from a blank cellular automaton canvas in the 'game of life').  

For instance, the pattern below can be generated from F, Right, Up, F, F, F, F, F.


Game Dev Notes: These games can be generated and filtered to interesting pictures (i.e. not blank) but right now the process is for the user to manually generate one.

Exploration note: Start with simple patterns, like a 2x2 block which just corresponds to a single "F" the hex command which translates to a 2x2 1111 block. Build to more complex.

Game Play Note: The website has a UI for drawing 2x2 blocks, but it seems like key strokes could be a better way of inputting values... it feels more like a coded spell than clicking a UI.

![draw pattern game](F_left_up_F_F_F_F_F.png)

#
Another Practical Approach to help RL learning that may be needed:  

1) Log human play -> 2) Supervised Learning Weight Initialization (mimic human actions) -> 3) Refine weights in Final stage of RL 

This approach is similar to the Alpha Go approach of training on top Go players before progressing to RL phase.

#

```
ca-rl v0.1.0
├── matplotlib v3.10.6
│   ├── contourpy v1.3.3
│   │   └── numpy v2.3.3
│   ├── cycler v0.12.1
│   ├── fonttools v4.60.1
│   ├── kiwisolver v1.4.9
│   ├── numpy v2.3.3
│   ├── packaging v25.0
│   ├── pillow v11.3.0
│   ├── pyparsing v3.2.5
│   └── python-dateutil v2.9.0.post0
│       └── six v1.17.0
├── numpy v2.3.3
├── scipy v1.16.2
│   └── numpy v2.3.3
├── tensorflow v2.20.0
│   ├── absl-py v2.3.1
│   ├── astunparse v1.6.3
│   │   ├── six v1.17.0
│   │   └── wheel v0.45.1
│   ├── flatbuffers v25.9.23
│   ├── gast v0.6.0
│   ├── google-pasta v0.2.0
│   │   └── six v1.17.0
│   ├── grpcio v1.75.1
│   │   └── typing-extensions v4.15.0
│   ├── h5py v3.14.0
│   │   └── numpy v2.3.3
│   ├── keras v3.11.3
│   │   ├── absl-py v2.3.1
│   │   ├── h5py v3.14.0 (*)
│   │   ├── ml-dtypes v0.5.3
│   │   │   └── numpy v2.3.3
│   │   ├── namex v0.1.0
│   │   ├── numpy v2.3.3
│   │   ├── optree v0.17.0
│   │   │   └── typing-extensions v4.15.0
│   │   ├── packaging v25.0
│   │   └── rich v14.1.0
│   │       ├── markdown-it-py v4.0.0
│   │       │   └── mdurl v0.1.2
│   │       └── pygments v2.19.2
│   ├── libclang v18.1.1
│   ├── ml-dtypes v0.5.3 (*)
│   ├── numpy v2.3.3
│   ├── opt-einsum v3.4.0
│   ├── packaging v25.0
│   ├── protobuf v6.32.1
│   ├── requests v2.32.5
│   │   ├── certifi v2025.8.3
│   │   ├── charset-normalizer v3.4.3
│   │   ├── idna v3.10
│   │   └── urllib3 v2.5.0
│   ├── setuptools v80.9.0
│   ├── six v1.17.0
│   ├── tensorboard v2.20.0
│   │   ├── absl-py v2.3.1
│   │   ├── grpcio v1.75.1 (*)
│   │   ├── markdown v3.9
│   │   ├── numpy v2.3.3
│   │   ├── packaging v25.0
│   │   ├── pillow v11.3.0
│   │   ├── protobuf v6.32.1
│   │   ├── setuptools v80.9.0
│   │   ├── tensorboard-data-server v0.7.2
│   │   └── werkzeug v3.1.3
│   │       └── markupsafe v3.0.3
│   ├── termcolor v3.1.0
│   ├── typing-extensions v4.15.0
│   └── wrapt v1.17.3
└── tqdm v4.67.1
```

# Command Line code 
Note: the code below is for the "pattern" game; games choices=['entropy', 'maxwell_demon', 'target_practice', 'pattern']  

Note: replace 'uv run' with 'python' or 'python3' in the code below if you are not using uv for virtual environment management. 
#


### Interactive Pattern Creator
#### Creates a custom_pattern .npy file for the 'pattern' game
### Controls:
Click cells to toggle them on/off"  
Press 's' to save pattern  
Press 'c' to clear grid  
Press 'q' to quit  
```
uv run rl_ca.py create_pattern
```
#

### Manual play 
#### Game Controls: 
4 arrow keys + 1 space bar + 16 hex keys (0-F) = 21 possible actions
#### Logging Controls:
'r' to record game play data  
'p' to save data immediately  
(Output: manual_data_<timestamp>.npz)
```
uv run rl_ca.py manual --reward pattern --pattern-file custom_pattern_12x12.npy
```
#

### Supervised Learning Pre-training 
#### Actor Policy Network & Critic Value Network are trained
```
uv run rl_ca.py supervised --data-file manual_data_1760318876.npz --epochs 40 --batch-size 64 --gamma 0.99 --value-coef 0.5
```
#

### Autonomous Demo (of Supervised Learning pre-trained model)
```
uv run rl_ca.py demo --weights supervised_weights_final.weights.h5 --reward pattern --pattern-file custom_pattern_12x12.npy
```
#

### RL Training (from scratch) 
show live plot results every 5 episodes
```
uv run rl_ca.py train --reward pattern --pattern-file custom_pattern_12x12.npy --live-plot 5
```

show every step of every episode (slow)
```
uv run rl_ca.py train --reward pattern --pattern-file custom_pattern_12x12.npy --live-plot 0
```
#

### RL Training (from Supervised Learning pre-training)
```
uv run rl_ca.py train --pretrained-weights supervised_weights_final.weights.h5 --reward pattern --pattern-file custom_pattern_12x12.npy --live-plot 2
```
#

### Autonomous Demo (of RL agent)
```
uv run rl_ca.py demo --weights ca_agent_weights_50.weights.h5 --reward pattern --pattern-file custom_pattern_12x12.npy
```
#

# six steps to get pattern
```
uv run rl_ca.py train --episodes 30 --rollout-steps 6 --live-plot 0 --reward pattern --pattern-file my_pattern.npy --pattern-file custom_pattern_12x12.npy
```

## Code Developed at Recurse Center
The [Recurse Center](https://www.recurse.com/) asks participants to program at the edge of their ability...

“The Edge... There is no honest way to explain it because the only people who really know where it is are the ones who have gone over. The others-the living-are those who pushed their control as far as they felt they could handle it, and then pulled back, or slowed down, or did whatever they had to when it came time to choose between Now and Later. But the edge is still Out there.”