import pytest
from src.utils.qccd_arch import QCCDArch
from src.utils.qccd_nodes import ManipulationTrap, QubitIon
from src.utils.qccd_operations import Operation, ParallelOperation, Split, Move, JunctionCrossing, Merge
from src.utils.qccd_operations_on_qubits import GateSwap, TwoQubitMSGate, OneQubitGate, Measurement, QubitReset
from src.compiler.qccd_parallelisation import (
    paralleliseOperationsSimple,
    paralleliseOperations,
    paralleliseOperationsWithBarriers,
    calculateDephasingFidelity,
    calculateDephasingFromIdling,
    happensBeforeForOperations,
)

QUBITS_PER_TRAP = 4

@pytest.fixture
def setup_arch():
    arch = QCCDArch()

    arch.SIZING = 0.5
    ions11 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap11 = arch.addManipulationTrap(x=0, y=0, ions=ions11)

    ions12 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap12 = arch.addManipulationTrap(x=1, y=0, ions=ions12)

    junctionL = arch.addJunction(x=0, y=0.5)
    junctionR = arch.addJunction(x=1, y=0.5)

    ions21 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap21 = arch.addManipulationTrap(x=0, y=1, ions=ions21)

    ions22 = [QubitIon() for _ in range(QUBITS_PER_TRAP)]
    trap22 = arch.addManipulationTrap(x=1, y=1, ions=ions22)

    crossing11 = arch.addEdge(trap11, junctionL)
    arch.addEdge(trap12, junctionR)
    crossing21 = arch.addEdge(trap21, junctionL)
    arch.addEdge(trap22, junctionR)
    arch.addEdge(junctionL, junctionR)

    arch.refreshGraph()

    ops = (
        Split.physicalOperation(trap11, crossing11), 
        Move.physicalOperation(crossing11),
        JunctionCrossing.physicalOperation(junctionL, crossing11),
        JunctionCrossing.physicalOperation(junctionL, crossing21),
        Move.physicalOperation(crossing21),
        Merge.physicalOperation(trap21, crossing21),
        GateSwap.physicalOperation(trap=trap21, ion1=ions21[0], ion2=ions21[2]),
        TwoQubitMSGate.physicalOperation(ion1=ions22[0], ion2=ions22[2],trap=trap22),
        OneQubitGate.physicalOperation(ion=ions12[0], trap=trap12),
        Measurement.physicalOperation(ion=ions12[0], trap=trap12),
        QubitReset.physicalOperation(ion=ions12[0],trap=trap12))
    schedule = (0,1,2,3,4,5,6,0,0,1,2)
    for op in ops:
        op.run()
    arch.refreshGraph()
    return ops, arch, schedule

      

def test_parallelise_operations(setup_arch):
    ops, _, schedule = setup_arch
    result = paralleliseOperations(ops)
    for op in ops:
        timestep = schedule[ops.index(op)]
        ops_before = [ops[i] for i in [j for j, t in enumerate(schedule) if t<timestep]]
        compiled_timestep = [k for k, os in result.items() if op in os.operations][0]
        for op_before in ops_before:
            if set(op_before.involvedComponents).intersection(op.involvedComponents):
                compiled_timestep_op_before = [k for k, os in result.items() if op_before in os.operations][0]
                assert compiled_timestep_op_before<compiled_timestep


def test_calculate_dephasing_fidelity():
    fidelity = calculateDephasingFidelity(1.0)
    assert 0.0 <= fidelity <= 1.0
    fidelity_long = calculateDephasingFidelity(1000.0)
    assert fidelity_long < fidelity  

def test_calculate_dephasing_from_idling(setup_arch):
    ops, _, _ = setup_arch
    result = calculateDephasingFromIdling(ops)
    assert isinstance(result, dict)
    for ion, dephasing_list in result.items():
        assert isinstance(ion, QubitIon)
        for op, fidelity in dephasing_list:
            assert isinstance(op, Operation)
            assert 0.0 <= fidelity <= 1.0
