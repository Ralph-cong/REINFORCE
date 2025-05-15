import gymnasium as gym
import os
import torch
import torch.nn.functional as F
from REINFORCE import PolicyNet  # 确保 PolicyNet 类在当前文件中或可导入

def eval(env_name="CartPole-v1", iter=3, model_path="./checkpoints/REINFORCE_CartPole-v1.pth", hidden_dim=128, device="cpu", render=True):
    """
    加载训练好的 REINFORCE 模型权重，并使用该策略玩游戏。

    Args:
        env_name (str): OpenAI Gym 环境的名称。
        model_path (str): 模型权重文件的路径。
        hidden_dim (int): 策略网络的隐藏层维度。
        device (str): 使用的设备 ("cpu" 或 "cuda").
        render (bool): 是否渲染游戏画面。
    """
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()  # 设置为评估模式

    for i in range(iter):
        episode_return = 0
        state, _ = env.reset()
        done = False
        step = 0
        print(f"Playing {env_name} with loaded policy...")
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = policy_net(state_tensor).cpu().squeeze(0).numpy() 

            action = probs.argmax() # 选择具有最高概率的动作

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            state = next_state
            step += 1
            if render:
                env.render()

        print(f"Episode finished after {step} timesteps. Total return: {episode_return:.2f}")
    env.close()

if __name__ == "__main__":
    # 确保模型权重文件存在于指定的路径
    model_path = "./checkpoints/REINFORCE_CartPole-v1.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weight file not found at {model_path}")
        print("Please run the training script first to generate the model weights.")
    else:
        eval(render=True,iter=3) # 设置 render=False 可以不显示游戏画面
