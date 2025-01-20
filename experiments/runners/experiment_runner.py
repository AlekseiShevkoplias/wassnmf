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
import argparse
from sklearn.metrics import pairwise_distances
from typing import Dict, Any

from wassnmf import WassersteinNMF

class ExperimentRunner:
    def __init__(self, output_dir: Path, chunk: Dict[str, Any] = None):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.results_file = self.output_dir / 'results.csv'
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )

    def _generate_data(self, n_features, n_samples, scenario):
        coord = np.linspace(-12, 12, n_features)
        X = np.zeros((n_features, n_samples))
        sigma = 1.0
        np.random.seed(42)
        
        if scenario == 'gaussian_mixture':
            for i in range(n_samples):
                X[:, i] = (
                    np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() + 6)) ** 2)
                    + np.random.rand() * np.exp(-(coord - sigma * np.random.randn()) ** 2)
                    + np.random.rand() * np.exp(-(coord - (sigma * np.random.randn() - 6)) ** 2)
                )
        else:  # simple test data
            X = np.random.rand(n_features, n_samples)
            
        # Normalize columns to simplex
        X /= X.sum(axis=0, keepdims=True)
        
        # Generate kernel matrix
        C = pairwise_distances(coord.reshape(-1, 1), metric="sqeuclidean")
        C /= np.mean(C)
        K = np.exp(-C / 0.025)
        
        return X, K

    def run_experiments(self):
        scenarios = [
            ('gaussian_mixture', 100, 100),
            ('gaussian_mixture', 200, 150),
        ]
        
        params_list = [
            {'n_components': nc, 'epsilon': eps, 'rho1': r1, 'rho2': r2, 'n_iter': ni}
            for nc in [2, 3, 4]
            for eps in [0.01, 0.025, 0.05]
            for r1 in [0.01, 0.05]
            for r2 in [0.01, 0.05]
            for ni in [10, 20]
        ]
        
        results = []
        for scenario_name, n_features, n_samples in scenarios:
            X, K = self._generate_data(n_features, n_samples, scenario_name)
            
            for params in params_list:
                try:
                    logging.info(f"Running experiment: {scenario_name} with params {params}")
                    
                    # Run WassNMF
                    start_time = time.time()
                    wnmf = WassersteinNMF(**params, verbose=False)
                    D_wass, Lambda_wass = wnmf.fit_transform(X, K)
                    wass_time = time.time() - start_time
                    
                    # Run standard NMF for comparison
                    nmf = NMF(n_components=params['n_components'], init='random', random_state=42)
                    W_standard = nmf.fit_transform(X)
                    H_standard = nmf.components_
                    
                    # Compute errors
                    wass_error = mean_squared_error(X, D_wass @ Lambda_wass)
                    standard_error = mean_squared_error(X, W_standard @ H_standard)
                    
                    # Save results
                    result = {
                        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
                        'scenario': scenario_name,
                        'n_features': n_features,
                        'n_samples': n_samples,
                        **params,
                        'wass_error': wass_error,
                        'standard_error': standard_error,
                        'time_taken': wass_time
                    }
                    results.append(result)
                    
                    # Save to CSV after each experiment
                    pd.DataFrame(results).to_csv(self.results_file, index=False)
                    
                    # Basic plotting
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    for i in range(D_wass.shape[1]):
                        ax1.plot(range(n_features), D_wass[:, i], label=f"Component {i+1}")
                    ax1.set_title(f"WassNMF Components\nError: {wass_error:.4f}")
                    ax1.legend()
                    
                    W_normalized = W_standard / W_standard.sum(axis=0)
                    for i in range(W_normalized.shape[1]):
                        ax2.plot(range(n_features), W_normalized[:, i], label=f"Component {i+1}")
                    ax2.set_title(f"Standard NMF\nError: {standard_error:.4f}")
                    ax2.legend()
                    
                    plt.savefig(self.figures_dir / f"comparison_{scenario_name}_{time.strftime('%H%M%S')}.png")
                    plt.close()
                    
                    logging.info(f"Experiment completed successfully")
                    
                except Exception as e:
                    logging.error(f"Error in experiment: {e}")
                    continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    runner.run_experiments()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()