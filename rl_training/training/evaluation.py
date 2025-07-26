"""
Evaluation utilities for Ardupilot RL training.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from ..environments import BaseEnv
from ..agents import BaseAgent


class Evaluator:
    """
    Evaluator for trained RL agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.n_eval_episodes = config.get('n_eval_episodes', 10)
        self.deterministic = config.get('deterministic', True)
    
    def evaluate_agent(self, agent: BaseAgent, env: BaseEnv) -> Dict[str, Any]:
        """
        Evaluate an agent on the given environment.
        
        Args:
            agent: Trained agent to evaluate
            env: Environment to evaluate on
            
        Returns:
            Dictionary containing evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        for episode in range(self.n_eval_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action = agent.predict(obs, deterministic=self.deterministic)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            episode_successes.append(not done)  # Success if not crashed
        
        # Compute statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Evaluation results dictionary
        """
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Number of episodes: {self.n_eval_episodes}")
        print(f"Deterministic: {self.deterministic}")
        print()
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print("=" * 50)
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            filepath: Path to save results
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")
    
    def compare_agents(self, agents: Dict[str, BaseAgent], env: BaseEnv) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple agents on the same environment.
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            env: Environment to evaluate on
            
        Returns:
            Dictionary mapping agent names to evaluation results
        """
        results = {}
        
        for agent_name, agent in agents.items():
            print(f"Evaluating {agent_name}...")
            agent_results = self.evaluate_agent(agent, env)
            results[agent_name] = agent_results
            self.print_results(agent_results)
            print()
        
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Plot evaluation results.
        
        Args:
            results: Evaluation results dictionary
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Reward distribution
            axes[0, 0].hist(results['episode_rewards'], bins=10, alpha=0.7)
            axes[0, 0].set_title('Episode Rewards Distribution')
            axes[0, 0].set_xlabel('Reward')
            axes[0, 0].set_ylabel('Frequency')
            
            # Episode lengths
            axes[0, 1].hist(results['episode_lengths'], bins=10, alpha=0.7)
            axes[0, 1].set_title('Episode Lengths Distribution')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Frequency')
            
            # Reward over episodes
            axes[1, 0].plot(results['episode_rewards'])
            axes[1, 0].set_title('Reward per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Reward')
            
            # Success rate
            success_rate = np.cumsum(results['episode_successes']) / np.arange(1, len(results['episode_successes']) + 1)
            axes[1, 1].plot(success_rate)
            axes[1, 1].set_title('Cumulative Success Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available. Install matplotlib to plot results.") 