# dEngine ucimlrepo extension

The following repo aims at extending [dEngine](https://github.com/DecentralizedLearning/dEngine) with the [UCI datasets](https://github.com/uci-ml-repo/ucimlrepo/tree/main).

It plugs into dEngine's usual config system (`BUILTINS`, `load_experiment_from_yamls`, etc.) and adds:
- dataset targets for datasets hosted on the UCI ML Repository, wrapped so they can be used as regular dEngine `dataset` config entries
- matching model architectures where needed
- ready-to-use builtin configs (`uci_BUILTINS`) that bundle a dataset + model + sensible default training engine, so they can be dropped straight into a dEngine experiment config list

## Installation
```
pip install https://github.com/DecentralizedLearning/dengine-uciml
```

## Available builtins

Builtins live under `dengine_uciml.configs.builtins.BUILTINS`, mirroring the structure of dEngine's own `BUILTINS`.

### `uci_BUILTINS.DATASETS.HAR_US`

Human Activity Recognition (HAR) dataset from smartphone sensor data, US variant. Config file: [`configs/datasets/HAR-US.yml`](configs/datasets/HAR-US.yml).

## Usage

A fully working example is provided in `main.py`. The important steps:

1. Import the plugin's builtins alongside dEngine's core ones:

   ```python
   from dengine.config.builtins import BUILTINS
   from dengine_uciml.configs.builtins import BUILTINS as uci_BUILTINS
   ```

2. Compose a config file list by mixing dEngine core configs (scenario, graph, partitioning) with a `uci_BUILTINS` dataset entry, exactly as you would with any other dEngine dataset builtin:

   ```python
   FEDERATED_CONFIGS = [
       BUILTINS.CORE.SCENARIOS.DECENTRALIZED_HOMOGENOUS,
       BUILTINS.CORE.GRAPH.STAR_51,
       BUILTINS.CORE.PARTITIONING.NONIID,
       uci_BUILTINS.DATASETS.HAR_US,
   ]
   ```

3. Load and run as usual with `load_experiment_from_yamls` / `load_engine`:

   ```python
   experiment = load_experiment_from_yamls(
       files=FEDERATED_CONFIGS,
       overrides=convert_to_nested_dict({
           "name": "cifar10,FedAvg",
           # any client/training_engine/etc. overrides
       }),
       experiments_directory_root=str(simulation_args.output_directory.absolute()),
       seed=simulation_args.seed,
   )
   ```

Because `uci_BUILTINS.DATASETS.HAR_US` already ships a `client.local_model` and `client.training_engine` block, you don't need to redefine those unless you want to override them (e.g. switching `client.target` to `DecAvgClient` for decentralized runs, as `main.py` does).
