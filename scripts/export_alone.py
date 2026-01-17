#!/usr/bin/env python3

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import torch first
import torch
import torch.nn as nn
import copy


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        return nn.ELU()


class ActorCriticMinimal(nn.Module):
    """Minimal ActorCritic for loading checkpoints."""
    
    def __init__(self, num_obs, num_privileged_obs, num_obs_history, num_actions):
        super().__init__()
        activation = get_activation('elu')
        
        # Encoder
        encoder_layers = [
            nn.Linear(num_privileged_obs, 256), activation,
            nn.Linear(256, 128), activation,
            nn.Linear(128, 18)
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Adaptation module
        adaptation_layers = [
            nn.Linear(num_obs_history, 256), activation,
            nn.Linear(256, 32), activation,
            nn.Linear(32, 18)
        ]
        self.adaptation_module = nn.Sequential(*adaptation_layers)
        
        # Actor
        actor_layers = [
            nn.Linear(18 + num_obs, 512), activation,
            nn.Linear(512, 256), activation,
            nn.Linear(256, 128), activation,
            nn.Linear(128, num_actions)
        ]
        self.actor_body = nn.Sequential(*actor_layers)
        
        # Critic
        critic_layers = [
            nn.Linear(18 + num_obs, 512), activation,
            nn.Linear(512, 256), activation,
            nn.Linear(256, 128), activation,
            nn.Linear(128, 1)
        ]
        self.critic_body = nn.Sequential(*critic_layers)


class PolicyExporter(torch.nn.Module):
    """Exports inference policy combining adaptation module and actor."""
    
    def __init__(self, actor_critic, num_obs):
        super().__init__()
        self.adaptation_module = copy.deepcopy(actor_critic.adaptation_module)
        self.actor_body = copy.deepcopy(actor_critic.actor_body)
        self.num_obs = num_obs
        
    def forward(self, obs_history):
        """
        Forward pass with single input (obs_history).
        Extracts current obs from history and passes both to actor.
        
        Args:
            obs_history: Full observation history [batch, num_obs_history]
            
        Returns:
            actions: Action tensor [batch, num_actions]
        """
        # Extract current observation (LAST num_obs elements, as history is [old->new])
        obs = obs_history[:, -self.num_obs:]
        latent = self.adaptation_module(obs_history)
        actions = self.actor_body(torch.cat((obs, latent), dim=-1))
        return actions


def infer_model_dims(checkpoint):
    """Infer model dimensions from checkpoint weights."""
    adaptation_weight = checkpoint['adaptation_module.0.weight']
    actor_weight = checkpoint['actor_body.0.weight']
    encoder_weight = checkpoint['encoder.0.weight']
    
    num_obs_history = adaptation_weight.shape[1]
    
    # Find actual latent dimension (last layer of adaptation module)
    max_adaptation_layer = max([int(k.split('.')[1]) for k in checkpoint.keys() 
                                if k.startswith('adaptation_module.') and '.weight' in k])
    latent_dim = checkpoint[f'adaptation_module.{max_adaptation_layer}.weight'].shape[0]
    
    num_obs = actor_weight.shape[1] - latent_dim
    num_privileged_obs = encoder_weight.shape[1]
    
    # Find num_actions (last layer of actor)
    max_actor_layer = max([int(k.split('.')[1]) for k in checkpoint.keys() 
                          if k.startswith('actor_body.') and '.weight' in k])
    num_actions = checkpoint[f'actor_body.{max_actor_layer}.weight'].shape[0]
    
    return num_obs, num_privileged_obs, num_obs_history, num_actions


def export_policy(checkpoint_path, output_path):
    """Export policy from checkpoint to JIT-traced .pt file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    num_obs, num_privileged_obs, num_obs_history, num_actions = infer_model_dims(checkpoint)
    
    print(f"Model dimensions: obs={num_obs}, privileged_obs={num_privileged_obs}, "
          f"obs_history={num_obs_history}, actions={num_actions}")
    
    actor_critic = ActorCriticMinimal(num_obs, num_privileged_obs, num_obs_history, num_actions)
    model_dict = actor_critic.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    actor_critic.load_state_dict(model_dict)
    actor_critic.eval()
    
    exporter = PolicyExporter(actor_critic, num_obs)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    exporter.to('cpu')
    traced_script_module = torch.jit.script(exporter)
    traced_script_module.save(output_path)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"✓ Policy exported to: {output_path} ({file_size:.2f} KB)")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--output', type=str, default='policy.pt', help='Output path')
    args = parser.parse_args()
    
    export_policy(args.checkpoint, args.output)
