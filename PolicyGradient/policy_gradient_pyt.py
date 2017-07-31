import nets.policy_net as PG
import gym
import numpy as np
from torch.autograd import Variable
import torch
import torch.optim as optim

torch.manual_seed(2017)


def discounted_reward(rewards, factor=.99):
    """
    compute the discounted reward of the (s, a) pair
    :param rewards: 1-D numpy.ndarray
    :param factor: float
    :return:
    """
    assert isinstance(rewards, list)
    rewards = np.array(rewards)
    discounted_r = np.zeros_like(rewards)
    running_add = 0.
    for t in reversed(range(rewards.size)):
        if rewards[t] != 0: running_add = 0  # if the reward!=0 the game will be reset.
        running_add = running_add * factor + rewards[t]
        discounted_r[t] = running_add
    return discounted_r  # the discounted reward per time step


def prepro(image):
    """
    preprocess the observe of the agent
    :param image: [210, 160, 3]
    :return: [80, 80, 3]
    """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image -= np.mean(image)
    image /= np.std(image)
    return image


def get_current_observation(current_img, prev_img):
    return current_img - prev_img if prev_img is not None else np.zeros_like(current_img, dtype=np.float32)


def do_backprop(observations, fake_grads, rewards, policy_net):
    """
    back-propagation
    :param observations: a list of states
    :param fake_grads: fake grads
    :param rewards: rewards
    :return:
    """
    # weight_decay_list = [child.weight for child in policy_net.children()]
    # bias_list = [child.bias for child in policy_net.children()]
    # opt = optim.SGD([
    #     {'params': weight_decay_list},
    #     {'params': bias_list}
    # ], lr=0.00005)

    assert isinstance(observations, list)
    assert isinstance(observations[0], np.ndarray)
    assert len(observations) == len(fake_grads) == len(rewards)
    observations = Variable(torch.FloatTensor(np.array(observations)), requires_grad=False).cuda()

    ## normalize discounted rewards
    discounted_r = discounted_reward(rewards, factor=.99)
    discounted_r = discounted_r - np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    back_grads = Variable(torch.FloatTensor(discounted_r), requires_grad=False).cuda() * Variable(
        torch.FloatTensor(np.array(fake_grads)), requires_grad=False).cuda()
    res = policy_net.forward(observations)
    res.backward(back_grads.view(-1, 1).data)
    policy_net.weight_decay_loss().backward()
    policy_net.get_optimizer().step()


def main():
    render = True
    # using to record the the observation and rewards within one episode
    observations = []  # store the observations
    rewards = []  # store the rewards used for compute discounted reward
    fake_grads = []  # fake rewards

    prev_img = None
    env = gym.make('Pong-v0')
    current_img = env.reset()

    policy_net = PG.PolicyNet()
    policy_net.cuda()

    GLOBAL_STEP = 0
    while True:
        if render:
            env.render()
        current_img = prepro(current_img)
        observation = get_current_observation(current_img, prev_img)
        observations.append(observation)
        prev_img = current_img

        ## Variable's World.
        policy_net_input = Variable(torch.FloatTensor(np.expand_dims(observation, axis=0)), volatile=True).cuda()

        action_prob = policy_net(policy_net_input)
        action = torch.bernoulli(action_prob)

        # we use the policy network to output the probability of take action 2
        # cause pytorch do the gradient descent, and we want gradient ascent
        # so the fake grad for the action up is -1 .
        if action.cpu().data.numpy()[0] == 1.:
            fake_grad = -1.
        else:
            fake_grad = 1.
        # fake_grad = action - action_prob  # fake gradient, improve the action you take
        # store the fake_grad using to do back-propagation
        fake_grads.append(fake_grad)

        # 2 up , 3 down
        which_action = (action.cpu() == Variable(torch.FloatTensor([1.])))
        action_you_gonna_take = 2 if which_action.data.numpy()[0] == 1  else 3
        current_img, reward, done, info = env.step(action_you_gonna_take)

        rewards.append(reward)

        if done == True or len(fake_grads) > 1050:
            GLOBAL_STEP += 1
            print("STEP: %d, the steps within one episode %d, mean rewards is %.7f" % (
                GLOBAL_STEP, len(rewards), np.mean(np.array(rewards))))
            arr_rewards = np.array(rewards)
            print("bonus--> computer %d : agent %d" % (np.sum(arr_rewards == -1.), np.sum(arr_rewards == 1.)))
            do_backprop(observations, fake_grads, rewards, policy_net)


            ## reset some buffers for the next episode
            observations = []
            fake_grads = []
            rewards = []
            prev_img = None
            current_img = env.reset()

    pass


if __name__ == '__main__':
    main()
