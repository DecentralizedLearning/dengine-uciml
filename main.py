from typing import Tuple
import re
from argparse import ArgumentParser
import signal
import torch.multiprocessing as mp
import json

from tqdm import tqdm
from pydantic import ValidationError

from dengine.bin.args_parser import cli_argument_parser
from dengine.config.builtins import BUILTINS
from dengine.bin import SimulationArguments
from dengine.config import ExperimentConfiguration
from dengine.config.utils import convert_to_nested_dict
from dengine import load_experiment_from_yamls
from dengine.bin.simulation import load_engine
from dengine.scenarios.decentralized import DecAvgClient
from dengine.scenarios.federated import FederatedClient
from dengine.scenarios.centralized import CentralizedClient

from dengine_uciml.configs.builtins import BUILTINS as uci_BUILTINS


def load_centralized_har_us(
    simulation_args: SimulationArguments,
    config_overrides,
):
    return [
        load_experiment_from_yamls(
            files=[
                BUILTINS.CORE.GRAPH.CENTRALIZED_GRAPH,
                BUILTINS.CORE.SCENARIOS.CENTRALIZED,
                uci_BUILTINS.DATASETS.HAR_US,
            ],
            overrides=convert_to_nested_dict({
                "name": "har_us,Centralized",
                "client.target": CentralizedClient,
                "client.training_engine.arguments.epochs": 100,
                **config_overrides,
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        )
    ]


def load_har_us_decentralized(
    simulation_args: SimulationArguments,
    config_overrides,
):
    return [
        load_experiment_from_yamls(
            files=[
                BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
                BUILTINS.CORE.GRAPH.BA_MEDIUM,
                uci_BUILTINS.DATASETS.HAR_US,
            ],
            overrides=convert_to_nested_dict({
                "name": "har_us,BA,DecAvg",
                "client.target": DecAvgClient,
                "graph.arguments.n": 21,
                "graph.arguments.seed": simulation_args.seed,
                "client.arguments.use_weighted_avg": True,
                **config_overrides,
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[
                BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
                BUILTINS.CORE.GRAPH.ER_MEDIUM,
                uci_BUILTINS.DATASETS.HAR_US,
            ],
            overrides=convert_to_nested_dict({
                "name": "har_us,ER,DecAvg",
                "client.target": DecAvgClient,
                "graph.arguments.n": 21,
                "graph.arguments.seed": simulation_args.seed,
                "client.arguments.use_weighted_avg": True,
                **config_overrides,
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        ),
        load_experiment_from_yamls(
            files=[
                BUILTINS.CORE.SCENARIOS.FEDERATED,
                BUILTINS.CORE.GRAPH.COMPLETE_GRAPH_MEDIUM,  # Topology is totally ignored in federated scenario
                uci_BUILTINS.DATASETS.HAR_US,
            ],
            overrides=convert_to_nested_dict({
                "name": "har_us,FedAvg",
                "client.target": FederatedClient,
                "graph.arguments.n": 21,
                **config_overrides,
            }),
            experiments_directory_root=str(simulation_args.output_directory.absolute()),
            seed=simulation_args.seed,
        ),
    ]


# ..... ..... ..... ..... ..... ..... ..... ..... #
# MAIN
# ..... ..... ..... ..... ..... ..... ..... ..... #
def _worker_init():
    """Ignore SIGINT in workers — let the parent process handle shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_config(
    args_tuple: Tuple[SimulationArguments, ExperimentConfiguration]
):
    simulation_args, cfg = args_tuple
    try:
        loaded_engine = load_engine(simulation_args, cfg, verbose=False)
        loaded_engine.run()
    except Exception as e:
        raise RuntimeError(f"Config {cfg} failed: {e}") from e


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--concurrent_runs",
        type=int,
        default=1,
        help="Number of configurations to run in parallel (default: 1)."
    )
    parser.add_argument(
        "--experiment_name_filter",
        type=str,
        action="append",
        default=None,
        help="Filter experiments names with a regex."
    )
    parser.add_argument(
        "--num_parallel_clients",
        type=int,
        default=1,
        help="Number of clients to run in parallel in each simulation (default: 1)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Sets the argument client.arguments.debug_skip_training=true"
    )
    simulation_args, _ = cli_argument_parser(parser)
    extra_arguments = parser.parse_args()

    parallelism_override = {}
    if extra_arguments.num_parallel_clients > 1:
        parallelism_override = {
            "scenario.arguments.num_parallel_clients": extra_arguments.num_parallel_clients,
        }
    debug_overrides = {}
    if extra_arguments.debug:
        debug_overrides = {
            "client.arguments.debug_skip_training": True,
            "client.arguments.debug_skip_test": True
        }

    configurations = []
    try:
        configurations += [
            *load_centralized_har_us(simulation_args, config_overrides=debug_overrides),
            *load_har_us_decentralized(
                simulation_args,
                config_overrides={**parallelism_override, **debug_overrides}),
        ]
    except ValidationError as e:
        print("\n❌ Failed to parse the configuration due to the following validation errors: ")
        for err in e.errors():
            loc = ".".join(str(part) for part in err['loc'])
            print(f" - {loc}: {err['msg']} (type: {err.get('type', 'unknown')})")
        return

    if extra_arguments.experiment_name_filter:
        selected_experiments = set()
        for pattern in extra_arguments.experiment_name_filter:
            regex_pattern = re.compile(pattern)
            selected_experiments.update(
                [cfg.name for cfg in configurations if regex_pattern.search(cfg.name)]
            )
        configurations = [cfg for cfg in configurations if cfg.name in selected_experiments]
        print(f"Found {len(configurations)} configs:")
        print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        try:
            input("Type [ENTER] to continue...")
        except KeyboardInterrupt:
            return

    if simulation_args.sanity_check:
        json_outfname = "experiments_configs.json"
        print(f'🟢 Configurations are fine. Dumped the following configs to {json_outfname}.')
        if not extra_arguments.experiment_name_filter:
            print(f"Found {len(configurations)} configs:")
            print("- " + "\n- ".join([cfg.name for cfg in configurations]))
        print("Ready to run...")
        all_configs = [cfg.model_dump() for cfg in configurations]
        with open(json_outfname, "w") as f:
            json.dump(all_configs, f, indent=2)
        return

    if len(configurations) == 1:
        try:
            loaded_engine = load_engine(simulation_args, configurations[0], verbose=False)
            loaded_engine.run()
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            raise RuntimeError(f"Config {configurations[0]} failed: {e}") from e

    pool_work_items = [(simulation_args, cfg) for cfg in configurations]
    try:
        with mp.Pool(processes=extra_arguments.concurrent_runs, initializer=_worker_init, maxtasksperchild=1) as pool:
            with tqdm(total=len(configurations), desc="Running configs", unit="cfg") as pbar:
                for _ in pool.imap_unordered(_run_config, pool_work_items):
                    pbar.update()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers.")
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
