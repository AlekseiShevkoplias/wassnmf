# experiments/runners/experiment_runner.py
import numpy as np
import pandas as pd
import time
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
import warnings
from sklearn.metrics import pairwise_distances
from itertools import product
from datetime import datetime
import traceback

from wassnmf.wassnmf import WassersteinNMF

class ExperimentRunner:
    def __init__(self, config_path: str, output_dir: str):
        # Load config
        with open(config_path) as f:
            self.config = json.load(f)
            
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )

    def _generate_data(self, scenario):
        n_features = scenario['n_features']
        n_samples = scenario['n_samples']
        coord = np.linspace(-12, 12, n_features)
        X = np.zeros((n_features, n_samples))
        sigma = 1.0
        
        if scenario['name'] == 'gaussian_mixture':
            for i in range(n_samples):
                X[:, i] = (
                    np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() + 6)) ** 2)
                    + np.random.rand() * np.exp(-(coord - sigma * np.random.randn()) ** 2)
                    + np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() - 6)) ** 2)
                )
        
        # Normalize to simplex
        X /= X.sum(axis=0, keepdims=True)
        
        # Generate kernel
        C = pairwise_distances(coord.reshape(-1, 1), metric="sqeuclidean")
        C /= np.mean(C)
        K = np.exp(-C / 0.025)
        
        return X, K, coord

    def run(self):
        """Run all experiments defined in config"""
        start_time = datetime.now()
        max_runtime = pd.Timedelta(hours=self.config['max_runtime_hours'])
        results = []
        experiment_count = 0
        
        try:
            # Generate all parameter combinations
            param_keys = self.config['params'].keys()
            param_values = self.config['params'].values()
            param_combinations = list(product(*param_values))
            
            total_experiments = len(self.config['scenarios']) * len(param_combinations)
            logging.info(f"Starting {total_experiments} experiments")
            
            # Main experiment loop
            for scenario in self.config['scenarios']:
                logging.info(f"Starting scenario: {scenario['name']}")
                
                # Generate data once per scenario
                X, K, coord = self._generate_data(scenario)
                
                for params in param_combinations:
                    # Check runtime
                    if datetime.now() - start_time > max_runtime:
                        logging.info("Maximum runtime reached, stopping")
                        break
                        
                    experiment_count += 1
                    param_dict = dict(zip(param_keys, params))
                    
                    try:
                        logging.info(f"Experiment {experiment_count}/{total_experiments}")
                        logging.info(f"Parameters: {param_dict}")
                        
                        # Run WassNMF
                        t0 = time.time()
                        wnmf = WassersteinNMF(**param_dict, verbose=True)
                        D_wass, Lambda_wass = wnmf.fit_transform(X, K)
                        wass_time = time.time() - t0
                        
                        # Run standard NMF
                        t0 = time.time()
                        nmf = NMF(n_components=param_dict['n_components'], init='random', random_state=42)
                        W_standard = nmf.fit_transform(X)
                        H_standard = nmf.components_
                        std_time = time.time() - t0
                        
                        # Compute errors
                        wass_error = mean_squared_error(X, D_wass @ Lambda_wass)
                        std_error = mean_squared_error(X, W_standard @ H_standard)
                        
                        # Save result
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'scenario': scenario['name'],
                            'n_features': scenario['n_features'],
                            'n_samples': scenario['n_samples'],
                            **param_dict,
                            'wass_error': wass_error,
                            'std_error': std_error,
                            'wass_time': wass_time,
                            'std_time': std_time
                        }
                        results.append(result)
                        
                        # Save results after each experiment
                        pd.DataFrame(results).to_csv(
                            self.output_dir / 'results.csv',
                            index=False
                        )
                        
                        # Plot comparison
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        for i in range(D_wass.shape[1]):
                            ax1.plot(coord, D_wass[:, i], label=f"Component {i+1}")
                        ax1.set_title(f"WassNMF (error: {wass_error:.4f})")
                        ax1.set_xlabel("Coordinate")
                        ax1.legend()
                        
                        for i in range(W_standard.shape[1]):
                            ax2.plot(coord, W_standard[:, i], label=f"Component {i+1}")
                        ax2.set_title(f"Standard NMF (error: {std_error:.4f})")
                        ax2.set_xlabel("Coordinate")
                        ax2.legend()
                        
                        plt.savefig(
                            self.figures_dir / f"comparison_{scenario['name']}_{experiment_count}.png"
                        )
                        plt.close()
                        
                        logging.info(f"Experiment {experiment_count} completed successfully")
                        logging.info(f"WassNMF error: {wass_error:.6f}, time: {wass_time:.2f}s")
                        logging.info(f"Standard error: {std_error:.6f}, time: {std_time:.2f}s")
                        
                    except Exception as e:
                        logging.error(f"Error in experiment {experiment_count}: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue
                        
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            logging.error(traceback.format_exc())
        
        finally:
            # Save final results
            if results:
                pd.DataFrame(results).to_csv(
                    self.output_dir / 'final_results.csv',
                    index=False
                )
            logging.info(f"Completed {experiment_count} experiments")

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "experiments/configs/params_20012025.json"
    output_dir = Path("wassnmf_experiments")
    
    runner = ExperimentRunner(config_path=config_path, output_dir=output_dir)
    runner.run()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()