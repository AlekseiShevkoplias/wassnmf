from pathlib import Path
import subprocess
import sys
import signal
import logging
import threading

def read_stream(stream, log_func):
    while True:
        line = stream.readline()
        if not line:
            break
        log_func(line.strip())

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

    process = None  # Initialize process outside the try block

    try:
        logging.info("Starting experiments")

        # Run the experiment runner, redirecting stderr to stdout
        process = subprocess.Popen(
            [sys.executable, str(runner_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, logging.info))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, logging.error))

        stdout_thread.start()
        stderr_thread.start()


        stdout_thread.join()
        stderr_thread.join()
        process.wait()  # Wait for the process to complete


        if process.returncode != 0:
            logging.error(f"Experiment failed with code: {process.returncode}")
            raise RuntimeError(f"Experiment failed with code: {process.returncode}") # raise error
        else:
            logging.info("Experiments completed successfully")

    except KeyboardInterrupt:
        logging.info("Received interrupt, stopping gracefully...")
        if process:  # Check if process was initialized
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.warning("Process did not terminate, sending SIGKILL...")
                try:
                    process.send_signal(signal.SIGKILL)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logging.error("Failed to kill process.")
                    raise
                except Exception as e:
                  logging.error(f"Error during forced termination: {e}")
                  raise

            except Exception as e:
                logging.error(f"Error stopping experiments: {e}")
                raise
    except Exception as e:
        logging.error(f"Error running experiments: {str(e)}")
        raise  # Re-raise the exception


if __name__ == "__main__":
    run_experiments()