
import numpy as np
from typing import (
    Sequence,
    List,
    Tuple,
    Mapping,
    Set,
    Dict,
)
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from collections import defaultdict, deque


def paralleliseOperationsSimple(
    operationSequence: Sequence[Operation],
) -> Sequence[ParallelOperation]:
    operationSequence = list(operationSequence)
    parallelOperationsSequence: List[ParallelOperation] = []
    if not operationSequence:
        return parallelOperationsSequence
    while operationSequence:
        parallelOperations = [operationSequence.pop(0)]
        involvedComponents: Set[QCCDComponent] = set(
            parallelOperations[0].involvedComponents
        )
        for op in operationSequence:
            components = op.involvedComponents
            if involvedComponents.isdisjoint(components):
                parallelOperations.append(op)
            involvedComponents = involvedComponents.union(components)
        for op in parallelOperations[1:]:
            operationSequence.remove(op)
        parallelOperation = ParallelOperation.physicalOperation(parallelOperations, [])
        parallelOperationsSequence.append(parallelOperation)
    return parallelOperationsSequence

def calculateDephasingFidelity(time: float) -> None:

    # # log(error) = m*log(delay)+c
    # m = (np.log(0.008)-np.log(0.00001))/((np.log(1)-np.log(0.01)))
    # c = np.log(0.00001)-m*np.log(0.01)
    # dephasingInFidelity = np.exp(m*np.log(time)+c)
    # # return 0.99999
    # return 1-dephasingInFidelity
    T2 = 2.2 # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    return 1 - (1-np.exp(-time/T2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330



def happensBeforeForOperations(
    operationSequence: Sequence[Operation], all_components: List[QCCDComponent]
) -> Tuple[Dict[Operation, List[Operation]], Sequence[Operation]]:
     # Step 1: Create a happens-before relation graph using adjacency list (DAG)
    happens_before: Dict[Operation, List[Operation]] = {op: [] for op in operationSequence} # Adjacency list for DAG
    indegree: Dict[Operation, int] = {op: 0 for op in operationSequence}  # Track number of dependencies for each operation
    operations_by_component: Dict[QCCDComponent, List[Operation]] = {c: [] for c in all_components}  # Track operations by QCCDComponent
    
    # Build the happens-before relation based on the components involved
    for op in operationSequence:
        for component in set(op.involvedComponents):
            for prev_op in operations_by_component[component]:
                # There is a happens-before relation (prev_op happens before op)
                if op not in happens_before[prev_op]:
                    happens_before[prev_op].append(op)
                    indegree[op] += 1
            operations_by_component[component].append(op)
    
    # Topologically sort the operations using Kahn's algorithm (BFS)
    zero_indegree_queue = deque([op for op in operationSequence if indegree[op] == 0])
    topologically_sorted_ops: List[Operation] = []
    while zero_indegree_queue:
        op = zero_indegree_queue.popleft()
        topologically_sorted_ops.append(op)
        for neighbor in happens_before[op]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree_queue.append(neighbor)
    return happens_before, topologically_sorted_ops

# Function to define a happens-before relation and schedule based on operation time
def paralleliseOperations(
    operationSequence: Sequence[Operation],
) -> Mapping[float, ParallelOperation]:
    all_components: List[QCCDComponent] = []
    for op in operationSequence:
        for c in op.involvedComponents:
            if c not in all_components:
                all_components.append(c)
    happens_before, topologically_sorted_ops = happensBeforeForOperations(operationSequence, all_components)
    topologically_sorted_ops = list(topologically_sorted_ops)
    # Schedule operations based on operation time
    time_schedule: Dict[float, List[Operation]] = defaultdict(list)  # Mapping from start time to a list of operations
    operation_end_times: Dict[Operation, float] = {}  # Store the end time for each operation
    component_busy_until: Dict[QCCDComponent, float] = {c: 0.0 for c in all_components}  # Track when each component is available
    current_time = 0.0
    earliest_start_times: Dict[Operation, float] = {op: 0.0 for op in operationSequence}
    parallel_operations: List[Operation] = []

    while topologically_sorted_ops:
        involved_components: Set[QCCDComponent] = set()
        operations_to_start: List[Operation] = []
        
        remaining_ops = topologically_sorted_ops[:]
        for op in remaining_ops:
            # check if the components are free at the required time
            component_ready_time = max(component_busy_until[comp] for comp in op.involvedComponents)
            start_time_for_op = max(component_ready_time, earliest_start_times[op])
            if start_time_for_op == current_time and involved_components.isdisjoint(op.involvedComponents):
                # If the operation can start now and its components are free
                operations_to_start.append(op)
                topologically_sorted_ops.remove(op)
                earliest_start_times.pop(op)
                operation_end_times[op] = current_time + op.operationTime()
                # Update when the components will be busy until
                for component in op.involvedComponents:
                    component_busy_until[component] = operation_end_times[op]
                for next_op in happens_before[op]:
                    earliest_start_times[next_op] = max(operation_end_times[op], earliest_start_times[next_op])
            involved_components.update(op.involvedComponents)
        if operations_to_start:
            # Assign the parallel operations to the current time slot
            time_schedule[current_time]=(ParallelOperation.physicalOperation(operations_to_start, parallel_operations))
            parallel_operations.extend(operations_to_start)
        # Move to the next available time (based on the minimum operation time)
        current_time = min([t for t in component_busy_until.values() if t>current_time])
        for op, op_end_time in operation_end_times.items():
            if op_end_time==current_time:
                parallel_operations.remove(op)
    return dict(time_schedule)

def paralleliseOperationsWithBarriers(
    operationSequence: Sequence[Operation],
    barriers: List[int]
) -> Mapping[float, ParallelOperation]:
    time_schedule = {}
    barriers.insert(0, 0)
    barriers.append(len(operationSequence))
    t=0.0
    for start, barrier in zip(barriers[:-1], barriers[1:]):
        for s, os in paralleliseOperations(operationSequence[start: barrier]).items():
            time_schedule[s+t] = os 
        t = max(x+max(y.operationTime() for y in ys.operations) for x, ys in time_schedule.items())
    return time_schedule

def calculateDephasingFromIdling(
    operationSequence: Sequence[Operation]
) -> Mapping[Ion, Sequence[Tuple[Operation, float]]]:
    all_components: List[QCCDComponent] = []
    for op in operationSequence:
        for c in op.involvedComponents:
            if c not in all_components:
                all_components.append(c)
    happens_before, topologically_sorted_ops = happensBeforeForOperations(operationSequence, all_components)
    topologically_sorted_ops = list(topologically_sorted_ops)
    # Schedule operations based on operation time
    operation_end_times: Dict[Operation, float] = {}  # Store the end time for each operation
    component_busy_until: Dict[QCCDComponent, float] = {c: 0.0 for c in all_components}  # Track when each component is available
    current_time = 0.0
    earliest_start_times: Dict[Operation, float] = {op: 0.0 for op in operationSequence}

    all_ions: List[Ion] = [c for c in all_components if isinstance(c, Ion)]
    ion_idling_times: Dict[Ion, List[Tuple[float, float]]] = {ion: [(0.0, 0.0)] for ion in all_ions}
    ion_idling_operations: Dict[Ion, List[QubitOperation]] = {ion: [] for ion in all_ions}
    idling_ions: List[Ion] = all_ions.copy()

    while True:
        involved_components: Set[QCCDComponent] = set()
        
        remaining_ops = topologically_sorted_ops[:]
        for op in remaining_ops:
            # check if the components are free at the required time
            component_ready_time = max(component_busy_until[comp] for comp in op.involvedComponents)
            start_time_for_op = max(component_ready_time, earliest_start_times[op])
            if start_time_for_op == current_time and involved_components.isdisjoint(op.involvedComponents):
                # If the operation can start now and its components are free
                topologically_sorted_ops.remove(op)
                earliest_start_times.pop(op)
                operation_end_times[op] = current_time + op.operationTime()
                # Pauli Z errors on ALL qubits during any crystal-reconfiguration operation, and on idle qubits not involved in an entangling gate 
                # so define idling qubits to be those not involved in a QubitOperation
                if isinstance(op, QubitOperation):
                    for ion in op.ions:
                        if ion in idling_ions:
                            idling_ions.remove(ion)
                            idling_start_time = ion_idling_times[ion][-1][0]
                            if current_time-idling_start_time>0:
                                ion_idling_times[ion][-1]=(idling_start_time, current_time-idling_start_time)
                                ion_idling_operations[ion].append(op)
                            else:
                                ion_idling_times[ion] = ion_idling_times[ion][:-1]
                # Update when the components will be busy until
                for component in op.involvedComponents:
                    component_busy_until[component] = operation_end_times[op]
                for next_op in happens_before[op]:
                    earliest_start_times[next_op] = max(operation_end_times[op], earliest_start_times[next_op])
            involved_components.update(op.involvedComponents)
        # Move to the next available time (based on the minimum operation time)
        next_time_steps = [t for t in component_busy_until.values() if t>current_time]
        if not next_time_steps:
            break
        current_time = min(next_time_steps)
        for op, op_end_time in operation_end_times.items():
            if op_end_time==current_time and isinstance(op, QubitOperation):
                for ion in op.ions:
                    if ion not in idling_ions:
                        idling_ions.append(ion)
                        ion_idling_times[ion].append((current_time, 0.0))
    # use calculateDephasingFidelity to find the dephasing noise (remove this method from Operation)
    ion_dephasing:  Dict[Ion, List[Tuple[Operation, float]]] = {ion: [] for ion in all_ions}
    for (ion, idling_times) in ion_idling_times.items():
        # do not include the last idling steps
        for k, (_, idling_duration) in enumerate(idling_times[:-1]):
            op_at_end_of_idle = ion_idling_operations[ion][k]
            ion_dephasing[ion].append((op_at_end_of_idle, calculateDephasingFidelity(idling_duration)))
    return ion_dephasing
