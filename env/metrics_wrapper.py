"""
Metrics tracking wrapper for FrozenLake.

This wrapper ensures that all environments (with or without reward shaping)
properly track:
- Original rewards
- Success indicators
- Episode metrics

This should be the OUTERMOST wrapper (applied last).
"""

import gymnasium as gym


class MetricsWrapper(gym.Wrapper):
    """
    Wrapper that adds consistent metric tracking to info dict.
    
    Ensures all environments return:
        info["orig_reward"]: The original environment reward
        info["is_success"]: Whether the goal was reached
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        """Reset environment."""
        state, info = self.env.reset(**kwargs)
        return state, info
    
    def step(self, action):
        """
        Step in environment and add metrics to info.
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Make info mutable
        info = dict(info)
        
        # If this is a reward shaping wrapper, orig_reward already exists
        # Otherwise, the reward IS the original reward
        if "orig_reward" not in info:
            info["orig_reward"] = reward
        
        # Detect success: reached goal (terminated with reward=1)
        # Use orig_reward to detect success, not shaped reward
        if "is_success" not in info:
            orig_reward = info.get("orig_reward", reward)
            info["is_success"] = bool(terminated and orig_reward == 1.0)
        
        return next_state, reward, terminated, truncated, info