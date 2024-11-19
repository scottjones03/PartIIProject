import pytest
from src.utils.qccd_arch import QCCDArch, QubitIon

@pytest.fixture
def qccd_arch():
    # Create architecture
    arch = QCCDArch()

    trap_spacing = 30
    traps = []

    for i in range(3):
        ions = [QubitIon('lightblue', 'Q') for _ in range(3)]
        trap = arch.addManipulationTrap(
            x=i * trap_spacing,
            y=0,
            ions=ions,
            color='grey',
            spacing=5,
            isHorizontal=True
        )
        traps.append(trap)

    crossings = []
    for t1, t2 in zip(traps[:-1], traps[1:]):
        crossings.append( arch.addEdge(t1, t2))

    return arch, traps, crossings

def test_architecture_creation(qccd_arch):
    _, traps, _ = qccd_arch

    for trap in traps:
        assert len(trap.ions) == 3

def test_edges_between_traps(qccd_arch):
    _, traps, crossings = qccd_arch

    for crossing, t1, t2 in zip(crossings, traps[:-1], traps[1:]):
        assert crossing.hasTrap(t1)
        assert crossing.hasTrap(t2)
