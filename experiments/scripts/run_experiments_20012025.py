# experiments/scripts/run_experiments_20012025.py
from pathlib import Path
import subprocess
import sys
import time
import signal
import logging

def run_experiments():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('supervisor.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    runner_script = repo_root / "experiments/runners/experiment_runner.py"
    
    try:
        logging.info("Starting experiments")
        
        # Run the experiment runner
        process = subprocess.Popen(
            [sys.executable, str(runner_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor output in real time
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            # Check if process is still running
            if process.poll() is not None:
                break
        
        # Get any remaining output
        _, stderr = process.communicate()
        if stderr:
            logging.error(f"Errors from experiment: {stderr}")
        
        if process.returncode != 0:
            logging.error(f"Experiment failed with code: {process.returncode}")
        else:
            logging.info("Experiments completed successfully")
            
    except KeyboardInterrupt:
        logging.info("Received interrupt, stopping gracefully...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        logging.error(f"Error running experiments: {str(e)}")
        raise

if __name__ == "__main__":
    run_experiments()