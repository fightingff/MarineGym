import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from marinegym.envs.isaac_env import AgentSpec, IsaacEnv
from marinegym.views import ArticulationView, RigidPrimView
from marinegym.utils.torch import euler_to_quaternion, quat_axis
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from .hover import Hover

class SwimToTarget(Hover):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        
        # 针对每个环境生成独立的目标位置
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.target_pos_dist = D.Uniform(
            torch.tensor([-3.0, -3.0, 1.0], device=self.device),
            torch.tensor([3.0, 3.0, 4.0], device=self.device)
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        # 调用父类的重置逻辑（包含流场、机器人姿态等）
        super()._reset_idx(env_ids)
        
        # 随机化每个环境的目标点
        new_targets = self.target_pos_dist.sample((len(env_ids), 1))
        self.target_pos[env_ids] = new_targets

        # 同步可视化目标的位置
        self.target_vis.set_world_poses(
            positions=new_targets.squeeze(1) + self.envs_positions[env_ids],
            env_indices=env_ids
        )

    def _compute_state_and_obs(self):
        # 重新计算相对位置（现在 target_pos 是 per-env 的）
        self.drone_state = self.drone.get_state()
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        obs = [self.rpos, self.drone_state[..., 3:], self.rheading,]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )
