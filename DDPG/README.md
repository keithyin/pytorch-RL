
**under construction, haven't finished yet**

# DDPG pytorch implentation

pytorch implementation of [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) .

**run model**

```shell
python train.py
```

## Model Detail

**model**


* optimizer: Adam

**behavior policy**

actor_net + ou_process

**target policy**

target_actor_net

**action-value function loss**

* MSE loss: doesn't work, don't know why
* Huber loss: worked


## Hyper Parameters

* replay buffer : 10e6
* ou_process
    * theta: 0.15
    * sigma: 0.2
* initialize:
    * xavier: doesn't work
    * the methods mentioned in the paper, works.
* value function:
    * BN : affine=True, doesn't work
    * without BN: worked
* moving average decay : 0.999(good, more stable), 0.99(bad)

## details
this program has been debugged 2 weeks, finally get it works, there is a lot of details to keep in mind:

* the parameter used to update target net
* parameter initialization method

