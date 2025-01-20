# experiments/scripts/run_experiments_20012025.py
import json
import sys
from pathlib import Path
import logging

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from experiments.runners.experiment_guardian import ExperimentGuardian

def main():
    # Load configuration
    config_path = repo_root / "experiments/configs/params_20012025.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiments with config: {config['experiment_name']}")
    
    # Initialize and run guardian
    guardian = ExperimentGuardian(
        max_runtime_hours=config['max_runtime_hours'],
        check_interval_seconds=config['system']['check_interval_seconds'],
        max_memory_percent=config['system']['max_memory_percent'],
        base_dir=config['system']['base_output_dir']
    )
    
    # Run experiments
    guardian.run()

if __name__ == "__main__":
    main()