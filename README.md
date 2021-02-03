# Deep Recurrent Q learning(DRQN) with Pytorch

Reference: https://arxiv.org/pdf/1507.06527.pdf

1. Pytorch(1.5.0)
2. Openai Gym(0.17.1)
3. Tensorboard (2.1.0)

## Training envrionment: OpenAI gym (CartPolev1)

&nbsp;
&nbsp;

<img src="./assets/cartpolev1.png" align="center"></img>

* * *
## POMDP

- CartPole-v1 environment is consists of Cart position/velocity, Pole angle/velocity.
&nbsp;

<img src="./assets/cartpolestate.png" align="center"></img>
- I set the partially observed state is position of cart and pole's angle. __The agent doesn't have idea of the velocity.__
* * *
## Stable Recurrent Updates
### 1. Bootstrapped Sequential Updates
- episodes are selected randomly from the replay memory and updates begin at the beginning of the episode and proceed forward through time to the conclusion of the episode. The targets at each timestep are generated from the target Q-network. The RNN's hidden state is carried forward throughout episode.
### 2. Sequential update
- Episodes are selected randomly from the replay memory and updates begin at random points in the episode and proceed for only unroll iterations timesteps(lookup_step). The targets at each timestep are generated from the target Q-network. __The RNN's initial state is zeroed at the start of the update.__

<img src="./assets/DRQN_param.png" align="center"></img>

- The above parameters are used to set the DRQN setting. __random update__ choose what update method to use.
- __lookup_step__ is how long step to observe. I found that longer lookup_step is better.
* * *
## DQN with Fully Oberserved vs DQN with POMDP vs DRQN with POMDP
&nbsp;
<img src="./assets/rewardlog.png" align="center" height="400px"></img>
- (orange)DQN with fully observed MDP situation can reach the highest reward.
- (blue)DQN with POMDP never can be reached to the high reward situation.
- (red)DRQN with POMDP can be reach the somewhat performance although it only can observe the position.


### TODO
- [x] Random update of DRQN
