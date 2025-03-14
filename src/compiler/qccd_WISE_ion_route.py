
from typing import (
    Sequence,
    List,
    Tuple,
    Set
)
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler._qccd_WISE_ion_routing import *
from src.compiler.qccd_qubits_to_ions import *

def ionRoutingWISEArch(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
) -> Tuple[Sequence[Operation], Sequence[int]]:
    allOps: List[Operation] = []
    barriers=[]
    operationsLeft = list(operations)
    while operationsLeft:


        # Run the single qubit operations that do not need routing

        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()
            for op in operationsLeft:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and trap and len(op.ions)==1:
                    op.setTrap(trap)
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(op.ions)

            for op in toRemove:
                op.run()
                allOps.append(op)
                operationsLeft.remove(op)
                
            if len(toRemove) == 0:
                break

        barriers.append(len(allOps))
      
        if not operationsLeft:
            break
        # Determine the operations that need routing
        toMove: List[TwoQubitMSGate] = []
        for op in operationsLeft:
            if isinstance(op, TwoQubitMSGate):
                toMove.append(op)

        # # Determine new global configuration
        ionsAdded = set()
        newArrangement = {trap: [] for trap in arch._manipulationTraps}
        toMoveCanDo: List[TwoQubitMSGate] = []

        trapIdx = 0
        for op in toMove:
            ion1, ion2 = op.ions
            ancilla, data = sorted(
                (ion1, ion2), key=lambda ion: ion.label[0]=='D'
            )
            if (ancilla.idx in ionsAdded) or (data.idx in ionsAdded):
                continue
            trap = data.parent
            if not isinstance(trap, ManipulationTrap):
                raise ValueError('Data Ion not in Trap!')
            
            trap = arch._manipulationTraps[trapIdx]
            if len(newArrangement[trap])+2>trap.capacity:
                trapIdx+=1
                if trapIdx == len(arch._manipulationTraps):
                    break
                trap = arch._manipulationTraps[trapIdx]
            newArrangement[trap].append(ancilla)
            newArrangement[trap].append(data)
            ionsAdded.add(ancilla.idx)
            ionsAdded.add(data.idx)
            toMoveCanDo.append(op)

        trapIdx = 0
        for trap in arch._manipulationTraps:
            for ion in trap.ions:
                if ion.idx not in ionsAdded:
                    trapIn = arch._manipulationTraps[trapIdx]
                    while trapIn.capacity<len(newArrangement[trapIn])+1:
                        trapIdx+=1
                        trapIn = arch._manipulationTraps[trapIdx]
                    newArrangement[trapIn].append(ion)
                    ionsAdded.add(ion.idx)
                    
        reconfig  = GlobalReconfigurations.physicalOperation(newArrangement, wiseArch)
       
        allOps.append(reconfig)
        reconfig.run()
        arch.refreshGraph()

        barriers.append(len(allOps))

        for op in toMoveCanDo:
            trap = op.getTrapForIons()
            op.setTrap(trap)
            op.run()
            allOps.append(op)
            operationsLeft.remove(op)

        barriers.append(len(allOps))
    return allOps, barriers


