import os
import argparse
import torch

parser = argparse.ArgumentParser(description="Options for running")
# Added type=int so you can compare option == 1 instead of "1"
parser.add_argument("--option", type=int, required=True, help="which configuration to use")
parser.add_argument("--debug", type=bool, required=False, default=False)
parser.add_argument("--gpu_num", type=str, required=False, default="3")
parser.add_argument("--seed", type=int, required=True, default="0")
parser.add_argument("--env", type=str, required=True, default="None")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

from envs.Navigation_Buttons_real import Navigation_Buttons_real

# from agents.GRPO_Predict_RWR import GRPO_Predict_RWR
from agents.uniform import uniform

# PyTorch
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

## JIT COntroller and Environment

# number of environments
env = Navigation_Buttons_real()

# num_envs = 2048
num_envs = 512

temp = 100
# number of epochs
epochs = 1001

device = torch.device("cpu")

d_conv = 4
d_v = 4
expand = 1
#
# hyperparmeters for transformer
num_heads = 4

max_steps=1000
num_envs = 1

## Training Loops
average_reward = torch.zeros(epochs, dtype=torch.float, device=device)
average_reward_shape = torch.zeros(epochs, dtype=torch.float, device=device)
max_reward = torch.zeros(epochs, dtype=torch.float, device=device)
satisfy = torch.zeros(epochs, dtype=torch.float, device=device)
prediction_error = torch.zeros(epochs, dtype=torch.float, device=device)

# transitions
# dvel = torch.tensor([-1, 0, 1, 0], device=device)
# dtheta = torch.tensor([0, torch.pi/2, 0, -torch.pi/2], device=device)

dvel = torch.tensor([-0.2, 0, 0.2, 0], device=device)
dtheta = torch.tensor([0, 0.2, 0, -0.2], device=device)

action_space = 2
num_actions = 4


agent = uniform()


# for t in tqdm(range(epochs)):
for e in range(epochs):

    # for calculation
    final_reward = torch.zeros(num_envs, device=device)
    rewards = torch.zeros(num_envs, max_steps, device=device)
    terminates = torch.zeros(num_envs, max_steps, device=device)

    i = 0

    actions = torch.zeros((num_envs, action_space), device=device)
    actions_agent = torch.zeros((num_envs, num_actions), device=device)

    # getting rid of inital values, before restting
    obs, reward, terminated, truncated, info = env.step(actions)
    obs, reward, terminated, truncated, info = env.step(actions)

    # reset
    agent.reset_state(num_envs)

    # experimenting without env.reset()
    obs, info = env.reset()
    
    no_action = torch.zeros(action_space, device=device)

    # target_radius = 2*2**(1/2)
    target_radius = 2**(1/2)

    base_speed = 0.5
    nominal_yaw = base_speed/target_radius
    # Kp = 0.1
    Kp = 0.1


    prev_error=0

    debug_action = 0

    # Step env
    with torch.no_grad():
        while not torch.all(terminated | truncated):

            hold = (terminated | truncated)

            # agent does not control second robot
            actions_agent[:, action] = 0

            action = agent.get_action(obs["policy"], actions_agent)

            # for learning
            actions_agent[:, action] = 1

            # for environment
            actions[:, 0] = dvel[action]
            # we don't want lateral movement
            actions[:, 2] = dtheta[action]

            actions[hold, :] = no_action
            # 

            # step
            obs, reward, new_terminated, truncated, info = env.step(actions)

            terminated = terminated | new_terminated

            final_reward += reward

            # rewards.append(reward)
            rewards[:, i] = reward
            terminates[:, i] = terminated
 
            i += 1


    # Final reward for the episode (per env)
    final_reward = (final_reward*(1-terminated.float())).detach() 

    # change this to be filtered by termination if you want
    rewards_shape = rewards*(1-terminated.unsqueeze(1).float())

    prediction_loss = agent.update(rewards_shape, terminates)

    # tracking
    average_reward[e] = final_reward.mean().item()
    average_reward_shape[e] = rewards_shape.sum(dim=1).mean().item()
    max_reward[e] = final_reward.max().item()
    satisfy[e] = (final_reward > 0).float().mean().item()
    prediction_error[e] = prediction_loss

    if e%20==0:
        print(f"epoch: {e+1}: average_reward: {average_reward[e]}, average_rewards_shape: {average_reward_shape[e]}, max_reward: {max_reward[e]}, meets: {satisfy[e]}, prediction_error: {prediction_error[e]}")


