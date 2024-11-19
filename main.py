import os
import yaml
import logging
import concurrent.futures
from typing import Any, Dict
from src.simulator.qccd_circuit import process_circuit
from datetime import datetime
import json  

def get_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def save_results(data: Dict[str, Any], output_dir: str):
    """Save experiment results to a timestamped JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)  
    output_path = os.path.join(output_dir, f"{timestamp}_experiment.json")

    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(f"Results saved to {output_path}")

def main(config_path: str):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    hardware = config["hardware"]
    qec = config["qec"]
    simulation = config["simulation"]

    distances = qec["distances"]
    capacities = hardware["trap_capacity"]
    gate_improvements = qec["gate_improvements"]
    num_shots = simulation["num_shots"]

    num_cores = hardware.get("num_cores", os.cpu_count())
    logger = get_logger(simulation["log_file"])
    data: Dict[str, Dict[str, Dict[int, Dict[int, Any]]]] = {
        "ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, 
        "QubitOperations": {}, "LogicalErrorRates": {}, 
        "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}
    }

    logger.info("Starting parallel processing of circuits")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_circuit, d, c, gate_improvements, num_shots)
            for d in distances for c in capacities
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                d = result["Distance"]
                c = result["Capacity"]
                for label in result["ElapsedTime"]:
                    data["ElapsedTime"][label][d][c] = result["ElapsedTime"][label]
                    data["Operations"][label][d][c] = result["Operations"][label]
                    data["MeanConcurrency"][label][d][c] = result["MeanConcurrency"][label]
                    data["QubitOperations"][label][d][c] = result["QubitOperations"][label]
                    data["LogicalErrorRates"][label][d][c] = result["LogicalErrorRates"][label]
                    data["PhysicalXErrorRates"][label][d][c] = result["PhysicalXErrorRates"][label]
                    data["PhysicalZErrorRates"][label][d][c] = result["PhysicalZErrorRates"][label]
                logger.info(f"Processed results for distance {d}, capacity {c}.")
            except Exception as e:
                logger.error("An error occurred during processing", exc_info=e)

    save_results(data, output_dir="data")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run QCCD circuit simulations.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
