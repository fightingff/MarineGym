import logging
import os
import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from marinegym import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from marinegym.learning import ALGOS

def load_checkpoint(checkpoint_path, env_config, algo_config):
    from marinegym.envs.isaac_env import IsaacEnv
    # 动态加载环境类
    env_name = env_config.task.name
    if env_name not in IsaacEnv.REGISTRY:
        raise ValueError(f"Task '{env_name}' not found in Registry. Available: {list(IsaacEnv.REGISTRY.keys())}")
    
    env_class = IsaacEnv.REGISTRY[env_name]
    base_env = env_class(env_config, headless=env_config.headless)
    
    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    
    # 初始化 Policy
    policy = ALGOS[algo_config.name.lower()](
        algo_config, 
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec, 
        device=base_env.device
    )
    
    # 加载权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=base_env.device)
    policy.load_state_dict(state_dict)
    
    return policy, env

import time
from torchrl.envs.utils import step_mdp

def evaluate_model(env, policy, num_episodes, cfg):
    results = []
    env.eval()
    env.set_seed(cfg.seed)
    
    # 如果不是 headless 模式，开启渲染
    if not cfg.headless:
        env.enable_render(True)

    # 从配置中获取可选的等待时间，默认为0.02秒，用于控制动画速度
    sleep_time = cfg.get("sleep_time", 0.02)
    max_steps = env.base_env.max_episode_length

    for i in tqdm(range(num_episodes), desc="Evaluating"):
        with set_exploration_type(ExplorationType.MODE):
            td = env.reset()
            episode_stats_list = []
            
            for _ in range(max_steps):
                # 1. 计算动作
                td = policy(td)
                # 2. 环境前进一步
                td = env.step(td)
                
                # 3. 如果开启了界面，加入延时以减缓动画速度
                if not cfg.headless:
                    time.sleep(sleep_time)
                
                # 保存统计信息 (如果有)
                if ("next", "stats") in td.keys(True, True):
                    episode_stats_list.append(td["next", "stats"].clone())

                # 如果所有环境都 done 了，就提前结束当前回合
                if td["next", "done"].all():
                    break
                    
                # 为下一步准备 td (将 next.observation 移到 observation 等)
                td = step_mdp(td)
            
            # 记录这一个 episode 最后一步的统计数据
            if episode_stats_list:
                last_stats = episode_stats_list[-1]
                # last_stats 通常有 num_envs 维度，我们取平均值或仅取第一个环境的指标
                results.append({k: v.item() if v.numel() == 1 else v.cpu().numpy() 
                              for k, v in last_stats.items()})
            
    return results

FILE_PATH = os.path.dirname(__file__)

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # 将 cfg 转换为可编辑字典，以支持在脚本中添加或修改键值
    OmegaConf.set_struct(cfg, False)
    
    # 允许从命令行覆盖 checkpoint 路径
    if "checkpoint" not in cfg:
        print("\n[Error] Please provide a checkpoint path.")
        print("Usage: python scripts/evaluate.py task=SwimToTarget +checkpoint=/path/to/model.pt\n")
        return

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    # 如果用户没有显式指定 headless，评估时默认开启 GUI
    if "headless" not in cfg or cfg.headless:
        cfg.headless = False 
    
    simulation_app = init_simulation_app(cfg)
    
    try:
        policy, env = load_checkpoint(cfg.checkpoint, cfg, cfg.algo)
        num_episodes = cfg.get("num_eval_episodes", 20)
        eval_results = evaluate_model(env, policy, num_episodes=num_episodes, cfg=cfg)
        
        # 统计并打印关键指标
        if eval_results:
            print("\n" + "="*40)
            print(f"Evaluation Summary ({num_episodes} episodes):")
            print(f"Task: {cfg.task.name}")
            print("-" * 40)
            
            # 自动提取所有标量指标并计算平均值
            metrics = eval_results[0].keys()
            for key in metrics:
                values = [res[key] for res in eval_results if isinstance(res[key], (int, float, np.number))]
                if values:
                    print(f"{key:<20}: {np.mean(values):.4f}")
            print("="*40 + "\n")
            
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
