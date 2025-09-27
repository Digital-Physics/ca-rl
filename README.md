This repository has a Python Version of Reinforcement Learning experimentation I'm working on.

The goal of the overall project is to have some RL agent learn to play a game in a cellular automata environment. I would like the RL training to engage the user, be visual, and be part of a distributed machine learning project. That's why it's being done in a website. And distributed study translates to client-side computation. TensorFlow.js was the choice for training in the browser (event though tensorflow.js is usually just used for inference in the browser) and will be used in the final implementation approach. But in the current phase, the exploratory phase of just trying to get RL to work, Python will be the language used, and this repo will be the source of the code.

The web page that uses tensorflow.js for RL training and visualization is here:
https://www.nets-vs-automata.net/rl.html

Once research is complete, the final configuration of the RL playground can be set-up again in tensorflow.js to keep the site a Distributed Learning site with ML/RL compute being being supplied by users on the client side.

Recent Updates:
-Ported RL for CA code from tensorflow.js on website to tensorflow Python for research & exploration.
--More accessible for other to explore code
--No need to focus on the UI aspects of the code
--Can train faster, without visualizations (That will be in final website version)

To do:
-Review code.
-Puffer lib possibilities 

Another Practical Approach to help RL learning:
Log human play -> Supervised Learning Weight Init. from human play -> Final stage of RL (Like Alpha Go approach of training on top Go players)


**To Train the Agent:**
Run the script from your terminal with the `train` command. You can customize the training parameters as needed.

# Train with default settings
python rl_ca.py train

# Train for more episodes with a different game mode
python rl_ca.py train --episodes 200 --reward maxwell_demon --grid-size 16

This will save the model weights (e.g., `ca_agent_weights_final.h5`) in the same directory.

# Watch the trained agent play
python rl_ca.py demo

# Play manually (just provide a dummy weights file name)
python rl_ca.py demo --weights no_weights.h5

When playing manually, make sure the plot window is active and use the `W/A/S/D` or arrow keys to move and the spacebar to pass a turn.



The Recurse Center asks us to program at the edge...

“The Edge... There is no honest way to explain it because the only people who really know where it is are the ones who have gone over. The others-the living-are those who pushed their control as far as they felt they could handle it, and then pulled back, or slowed down, or did whatever they had to when it came time to choose between Now and Later. But the edge is still Out there.”