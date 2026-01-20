- 0117
```txt
failed:
1. 更改日志为 tensorboard，迭代速度很慢
2. 将 ml_logger 替换为 tensorboard，出现问题

log:
1. 072748.649209 原始配置
2. 092449.713244 速度上限提高到 3.0 m/s
    2.2 后发散
```
```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/rapid-locomotion-rl/scripts/train.py
```

- 0119-1
```python
_ = Cnfg.rewards
_.only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
_.tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
_.tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
_.tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
_.tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
_.soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
_.soft_dof_vel_limit = 1.
_.soft_torque_limit = 1.
_.base_height_target = 0.31
_.max_contact_force = 100.  # forces above this value are penalized
_.use_terminal_body_height = False
_.terminal_body_height = 0.20

_ = Cnfg.rewards.scales
_.termination = -0.0
_.tracking_lin_vel = 1.0
_.tracking_ang_vel = 0.5
_.lin_vel_z = -2.0
_.ang_vel_xy = -0.05
_.orientation = -5.
_.torques = -0.0001
_.dof_vel = -0.
_.dof_acc = -2.5e-7
_.base_height = -30.
_.feet_air_time = 1.0
_.collision = -1.
_.feet_stumble = -0.0
_.action_rate = -0.01
_.stand_still = -0.
_.tracking_lin_vel_lat = 0.
_.tracking_lin_vel_long = 0.
_.dof_pos_limits = -10.0
# more
_.hip_pos0 = -0.05
_.feet_mirror = -0.1

_ = Cnfg.domain_rand
_.randomize_base_mass = True
_.added_mass_range = [-1, 2]
_.push_robots = True
_.max_push_vel_xy = 1
_.push_interval_s = 15
_.randomize_friction = True
_.friction_range = [0.2, 1.25]
_.randomize_restitution = True
_.restitution_range = [0.0, 1.0]
_.restitution = 0.5  # default terrain restitution
_.randomize_com_displacement = True
_.com_displacement_range = [-0.05, 0.05]
_.randomize_motor_strength = True
_.motor_strength_range = [0.9, 1.1]
_.randomize_Kp_factor = True
_.Kp_factor_range = [0.9, 1.1]
_.randomize_Kd_factor = True
_.Kd_factor_range = [0.9, 1.1]
_.rand_interval_s = 6
_.randomize_lag_timesteps = True
_.lag_timesteps = 6
```
```txt
name: mr_lag
```
