#!/usr/bin/env python3
"""
Evaluation demo for Ardupilot RL.

This example demonstrates how to evaluate trained models
and compare different agents.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rl_training.environments import ArdupilotEnv
from rl_training.agents import PPOAgent, SACAgent, TD3Agent
from rl_training.training import Evaluator
from rl_training.utils.utils import load_config


def main():
    """Run evaluation demo."""
    print("Starting Ardupilot RL evaluation demo...")
    
    # Load configuration
    config = load_config()
    
    # Create environment
    print("Creating Ardupilot environment...")
    env = ArdupilotEnv(config)
    
    # Create evaluator
    print("Setting up evaluator...")
    evaluator = Evaluator(config.get('evaluation_config', {}))
    
    # Create agents
    agents = {}
    
    # PPO Agent
    print("Loading PPO agent...")
    try:
        ppo_agent = PPOAgent(config)
        ppo_agent.load("./trained_model_ppo")
        agents["PPO"] = ppo_agent
    except:
        print("PPO model not found, skipping...")
    
    # SAC Agent
    print("Loading SAC agent...")
    try:
        sac_agent = SACAgent(config)
        sac_agent.load("./trained_model_sac")
        agents["SAC"] = sac_agent
    except:
        print("SAC model not found, skipping...")
    
    # TD3 Agent
    print("Loading TD3 agent...")
    try:
        td3_agent = TD3Agent(config)
        td3_agent.load("./trained_model_td3")
        agents["TD3"] = td3_agent
    except:
        print("TD3 model not found, skipping...")
    
    if not agents:
        print("No trained models found. Please train some models first.")
        return
    
    # Compare agents
    print("Comparing agents...")
    results = evaluator.compare_agents(agents, env)
    
    # Save results
    print("Saving evaluation results...")
    evaluator.save_results(results, "./evaluation_results.json")
    
    # Plot results for each agent
    for agent_name, agent_results in results.items():
        print(f"Plotting results for {agent_name}...")
        evaluator.plot_results(agent_results, f"./evaluation_plot_{agent_name.lower()}.png")
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main() 