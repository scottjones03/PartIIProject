import pytest
import numpy as np
from src.utils.qccd_nodes import QubitIon
from src.utils.qccd_arch import QCCDArch
from src.compiler.qccd_qubits_to_ions import regularPartition, arrangeClusters

@pytest.fixture
def create_ions():
    """Fixture to create ions with specific positions."""
    def _create_ions(positions):
        ions = [QubitIon() for _, _ in positions]
        for i, (x, y) in enumerate(positions):
            ions[i].set(ions[i].idx, x, y)
        return ions
    return _create_ions

def test_partition_cluster_ions(create_ions):
    dions = create_ions([(0, 0), (1, 1), (3, 3), (4, 4)])
    mions = create_ions([(0.5, 0.5), (3.5, 3.5)])
    trap_capacity = 4

    clusters = regularPartition(mions, dions, trap_capacity)
    for cluster, center in clusters:
        if dions[0] in cluster:
            assert center[0] == center[1] == 0.5
        if dions[-1] in cluster:
            assert center[0]==center[1]==3.5

def test_arrange_clusters(create_ions):
    grid_positions = [(0,0),(1,0),(0,1),(1,1)]
    clusters = [(create_ions([(i, j), (i, j)]), (i,j)) for i, j in grid_positions]
    arranged_positions = arrangeClusters(clusters, grid_positions)
    for (_, clpos), pos in zip(clusters, arranged_positions):
        assert clpos == pos
