from typing import Union

from params_proto import Meta

from mini_gym.envs.base.legged_robot_config import Cfg


def config_go1(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state

    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    _.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {'joint': 20.}  # [N*m/rad]
    _.damping = {'joint': 0.5}  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    _.action_scale = 0.25
    _.hip_scale_reduction = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    _.decimation = 4

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
    _.foot_name = "foot"
    _.penalize_contacts_on = ["thigh", "calf"]
    _.terminate_after_contacts_on = ["base"]
    _.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    _.flip_visual_attachments = False
    _.fix_base_link = False

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

    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50

    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False

    _ = Cnfg.env
    _.num_observations = 42
    _.observe_vel = False
    _.num_envs = 4096

    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]
    _.lin_vel_y = [-1.0, 1.0]

    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 10.0
    _.command_curriculum = True
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.6, 0.6]
    _.lin_vel_y = [-0.6, 0.6]
    _.ang_vel_yaw = [-1, 1]
    _.max_forward_curriculum = 3.

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