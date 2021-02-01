# Deep Recurrent Q learning(DRQN) with Pytorch

Reference: https://arxiv.org/pdf/1507.06527.pdf

## Training envrionment: OpenAI gym (CartPolev1)

<img src="./assets/cartpolev1.png" align="center"></img>
* * *
## POMDP
- CartPole-v1 environment is consists of Cart position/velocity, Pole angle/velocity.
<img src="./assets/cartpolestate.png" align="center"></img>
- I set the partially observed state is position of cart and pole's position. The agent doesn't have idea of the velocity.
* * *
## DQN with Fully Oberserved vs DQN with POMDP vs DRQN with POMDP
<img src="./assets/rewardlog.png" align="center"></img>
- DQN with fully observed MDP situation can reach the highest reward.
- DQN with POMDP never can be reached to the high reward situation.
- DRQN with POMDP can be reach the somewhat performance although it only can observe the position.


### TODO
- [ ] Random update of DRQN
