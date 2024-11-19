import pytest
from src.compiler.qccd_ion_routing import ionRouting
from src.simulator.qccd_circuit import QCCDCircuit

TEST_DISTANCE = 3
TEST_CAPACITY = 2

@pytest.fixture
def setup_circuit():
    """
    Generate a QCCDCircuit for testing.
    """

    circuit = QCCDCircuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=TEST_DISTANCE,
    )
    arch, (instructions, _) = circuit.processCircuitAugmentedGrid(
        rows=TEST_DISTANCE + 1, 
        cols=TEST_DISTANCE + 1, 
        trapCapacity=TEST_CAPACITY, 
    )
    arch.refreshGraph()
    return arch, instructions

def test_no_operations(setup_circuit):
    """
    Test `ionRouting` when no operations are provided.
    """
    arch, _ = setup_circuit
    operations = []

    allOps, barriers = ionRouting(arch, operations, TEST_CAPACITY)

    assert allOps == []
    assert barriers == []

def test_single_operation(setup_circuit):
    """
    Test `ionRouting` with a single operation.
    """
    arch, instructions = setup_circuit

    # Use the first operation only
    operations = [instructions[0]]

    allOps, barriers = ionRouting(arch, operations, TEST_CAPACITY)

    assert len(allOps) == 1
    assert barriers == [1]

def test_with_full_routing(setup_circuit):
    """
    Test `ionRouting` with a realistic set of instructions requiring routing.
    """
    arch, instructions = setup_circuit
   
    allOps, _ = ionRouting(arch, instructions, TEST_CAPACITY)
    for op in allOps:
        op.run()
        for node in arch.nodes.values():
            assert len(node.ions)<=node.capacity

    assert len(allOps) >= len(instructions)


def test_high_trap_capacity(setup_circuit):
    """
    Test `ionRouting` with high trap capacity.
    """
    trapCapacity = 2*TEST_DISTANCE**2+1
    circuit = QCCDCircuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=TEST_DISTANCE,
    )
    arch, (instructions, _) = circuit.processCircuitAugmentedGrid(
        rows=TEST_DISTANCE + 1, 
        cols=TEST_DISTANCE + 1, 
        trapCapacity=trapCapacity, 
    )
    arch.refreshGraph()
    
    allOps, _ = ionRouting(arch, instructions, trapCapacity)

    # High capacity means there should be no ion movements
    assert len(allOps) == len(instructions)
