import random

import torch

import dengine as d
from dengine.graph import Graph
from dengine.partitioning import TYPE_DATASET_PARTITIONING, split_partitions_into_train_and_test
from dengine.partitioning.decorators import register_partitioning

from dengine_uciml.dataset import SupervisedDataset


@register_partitioning()
def har_us_partition_by_subject(
    dataset: d.SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    assert isinstance(dataset, SupervisedDataset)
    assert len(graph.nodes) <= 30

    subjects = (dataset.subjects.unique()).tolist()
    random.shuffle(subjects)

    subjects_partitions = {}
    for sub, node in zip(subjects, graph.nodes):
        subjects_partitions[node] = torch.where(dataset.subjects == sub)[0]

    return split_partitions_into_train_and_test(validation_percentage, subjects_partitions)
