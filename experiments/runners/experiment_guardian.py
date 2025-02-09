import sys
import time
import signal
import logging
import traceback
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import psutil
from typing import Optional

class ExperimentGuardian:
    def __init__(
        self,
        max_runtime_hours: float = 7.0,
        check_interval_seconds: int = 300,
        max_memory_percent: float = 90.0,
        base_dir: str = "wassnmf_experiments"
    ):
        self.max_runtime = timedelta(hours=max_runtime_hours)
        self.check_interval = check_interval_seconds
        self.max_memory_percent = max_memory_percent
        self.base_dir = Path(base_dir)
        self.guardian_log = self.base_dir / 'guardian.log'
        self.status_file = self.base_dir / 'guardian_status.json'
        self.termination_flag = self.base_dir / 'terminate.flag'
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure guardian logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - GUARDIAN - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.guardian_log),
                logging.StreamHandler()
            ]
        )
        
        # Initialize status
        self.status = self._load_status()
        
    def _load_status(self) -> dict:
        """Load or initialize guardian status"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load status file: {e}")
        
        return {
            'start_time': datetime.now().isoformat(),
            'skipped_chunks': [],
            'completed_chunks': [],
            'current_chunk': None
        }
    
    def _save_status(self):
        """Save guardian status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def _check_system_resources(self) -> bool:
        """Check if system resources are within acceptable limits"""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.max_memory_percent:
                logging.warning(f"Memory usage too high: {memory.percent}%")
                return False
            
            # Check CPU temperature if available
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps and 'coretemp' in temps:
                    for temp in temps['coretemp']:
                        if temp.current > 85:  # CPU too hot
                            logging.warning(f"CPU temperature too high: {temp.current}°C")
                            return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking system resources: {e}")
            return True  # Continue if we can't check
    
    def _get_next_chunk(self) -> Optional[dict]:
        """Get the next experiment chunk to run"""
        # Define parameter chunks
        param_chunks = [
            {'n_components': 2, 'param_set': 1},
            {'n_components': 3, 'param_set': 1},
            {'n_components': 4, 'param_set': 1},
            {'n_components': 2, 'param_set': 2},
            {'n_components': 3, 'param_set': 2},
            {'n_components': 4, 'param_set': 2},
        ]
        
        for chunk in param_chunks:
            chunk_id = f"params_{chunk['n_components']}_{chunk['param_set']}"
            if (chunk_id not in self.status['completed_chunks'] and 
                chunk_id not in self.status['skipped_chunks']):
                return {'id': chunk_id, **chunk}
        
        return None

    def run_experiment_chunk(self, chunk: dict) -> bool:
        """Run a single chunk of experiments. Returns True if successful, False if failed/skipped."""
        try:
            # Create a new Python process
            repo_root = Path(__file__).resolve().parent.parent.parent
            runner_script = repo_root / "experiments/runners/experiment_runner.py"
            
            cmd = [
                sys.executable,
                str(runner_script),
                "--output-dir", str(self.base_dir),
                "--chunk", json.dumps(chunk)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor the process
            while process.poll() is None:
                # Check if we should terminate
                if self.termination_flag.exists():
                    process.terminate()
                    logging.info("Termination flag detected, stopping gracefully...")
                    return False
                
                # Check system resources
                if not self._check_system_resources():
                    process.terminate()
                    logging.warning("System resources critical, pausing...")
                    time.sleep(300)  # Cool-down period
                    return False
                
                time.sleep(self.check_interval)
            
            # Check if process completed successfully
            if process.returncode == 0:
                self.status['completed_chunks'].append(chunk['id'])
                self._save_status()
                return True
            else:
                logging.error(f"Chunk {chunk['id']} failed with return code {process.returncode}")
                self.status['skipped_chunks'].append(chunk['id'])
                self._save_status()
                return False
            
        except Exception as e:
            logging.error(f"Error running chunk {chunk['id']}: {e}")
            logging.error(traceback.format_exc())
            self.status['skipped_chunks'].append(chunk['id'])
            self._save_status()
            return False

    def run(self):
        """Main guardian loop"""
        logging.info("Starting experiment guardian...")
        start_time = datetime.now()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.termination_flag.touch())
        signal.signal(signal.SIGINT, lambda s, f: self.termination_flag.touch())
        
        try:
            while datetime.now() - start_time < self.max_runtime:
                # Get next chunk to run
                chunk = self._get_next_chunk()
                if not chunk:
                    logging.info("All experiments completed or skipped!")
                    break
                
                # Log current chunk
                self.status['current_chunk'] = chunk['id']
                self._save_status()
                
                # Run the chunk
                logging.info(f"Starting chunk: {chunk['id']}")
                success = self.run_experiment_chunk(chunk)
                
                if success:
                    logging.info(f"Chunk {chunk['id']} completed successfully")
                else:
                    logging.warning(f"Chunk {chunk['id']} failed/skipped, moving to next")
                
                # Small delay between chunks
                time.sleep(10)
            
        except Exception as e:
            logging.error(f"Fatal error in guardian: {e}")
            logging.error(traceback.format_exc())
        
        finally:
            # Final status update
            self.status['end_time'] = datetime.now().isoformat()
            self._save_status()
            logging.info("Guardian shutting down...")
            
            # Log summary
            n_completed = len(self.status['completed_chunks'])
            n_skipped = len(self.status['skipped_chunks'])
            logging.info(f"Run completed. Successfully finished {n_completed} chunks, "
                        f"skipped {n_skipped} chunks.")

if __name__ == "__main__":
    guardian = ExperimentGuardian(
        max_runtime_hours=7.0,
        check_interval_seconds=300,  # Check every 5 minutes
        max_memory_percent=90.0,
        base_dir="wassnmf_experiments"
    )
    guardian.run()