import gymnasium as gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
from REINFORCE import REINFORCE
import os
import shutil
from datetime import datetime
import numpy as np

def main():
    write = True
    save = True
    learning_rate = 1e-3
    num_episodes = 3000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                    device)
    if write:
        from torch.utils.tensorboard import SummaryWriter
        #Use SummaryWriter to record the trainig
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16]+ '_' + timenow[-2::]
        writepath = 'runs/{}'.format(env_name) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    return_list = []
    with tqdm(total=int(num_episodes), desc='Episode') as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state,_ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated,truncated, _ = env.step(action)
                done = terminated or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (i_episode+1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                writer.add_scalar('return', np.mean(return_list[-10:]),
                                  i_episode + 1)
                
            pbar.update(1)
    
    if save:
        save_dir = './checkpoints'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(agent.policy_net.state_dict(),
                os.path.join(save_dir, f'REINFORCE_{env_name}.pth'))
    env.close()

if __name__ == "__main__":
    main()
