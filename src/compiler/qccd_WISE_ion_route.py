
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



def ionRoutingWISEArch2(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
) -> Tuple[Sequence[Operation], Sequence[int]]:
    allOps: List[Operation] = []
    barriers: List[int] = []
    operationsLeft = list(operations)

    def current_trap(ion: Ion) -> ManipulationTrap:
        t = ion.parent
        if not isinstance(t, ManipulationTrap):
            raise ValueError("Ion not in ManipulationTrap")
        return t

    def move_cost(ion: Ion, target: ManipulationTrap) -> float:
        # plug in your real metric: junction hops, shuttles, etc.
        return (current_trap(ion).pos[0]-target.pos[0])**2+(current_trap(ion).pos[1]-target.pos[1])**2

    def odd_even_future_penalty(pair: Tuple[Ion, Ion], target: ManipulationTrap) -> float:
        # Look-ahead: where should these be next round under odd-even transposition?
        # Return 0 if target is consistent, small positive otherwise.
        a, d = pair
        should = (current_trap(d) if current_trap(d).numIons<current_trap(d).capacity else None)
        return 0.0 if should is None or should == target else 1.0

    while operationsLeft:

        # 1) Eagerly run commuting single-qubit ops with no routing.
        while True:
            toRemove: List[Operation] = []
            ionsBusy: Set[Ion] = set()
            for op in operationsLeft:
                if len(op.ions) != 1:
                    continue
                if not ionsBusy.isdisjoint(op.ions):
                    continue
                trap = op.getTrapForIons()
                if trap:
                    op.setTrap(trap)
                    toRemove.append(op)
                    ionsBusy |= op.ions
            if not toRemove:
                break
            for op in toRemove:
                op.run()
                allOps.append(op)
                operationsLeft.remove(op)

        barriers.append(len(allOps))
        if not operationsLeft:
            break

        # 2) Collect ready two-qubit gates.
        ready_MS: List[TwoQubitMSGate] = []
        ionsLocked: Set[int] = set()
        for op in operationsLeft:
            if isinstance(op, TwoQubitMSGate):
                i1, i2 = op.ions
                if (i1.idx not in ionsLocked) and (i2.idx not in ionsLocked):
                    ready_MS.append(op)
                    ionsLocked.add(i1.idx); ionsLocked.add(i2.idx)

        if not ready_MS:
            # nothing to route this round (deadlock safety)
            break

        # 3) Build gate–trap candidate edges with costs.
        traps: List[ManipulationTrap] = list(arch._manipulationTraps)
        edges: List[Tuple[float, int, int]] = []  # (cost, gate_idx, trap_idx)

        alpha, beta, gamma = 1.0, 0.3, -0.05
        for gi, g in enumerate(ready_MS):
            i1, i2 = g.ions
            # normalize: (ancilla, data) ordering if you like
            anc, data = sorted((i1, i2), key=lambda ion: ion.label[0] == 'D')
            for tj, T in enumerate(traps):
                if T.capacity < 2:
                    continue
                c = alpha * (move_cost(anc, T) + move_cost(data, T))
                c += beta * odd_even_future_penalty((anc, data), T)
                if current_trap(anc) == T or current_trap(data) == T:
                    c += gamma
                edges.append((c, gi, tj))

        # 4) Select a max-cardinality, min-cost assignment (greedy ok).
        edges.sort(key=lambda x: x[0])
        chosen_gates: Set[int] = set()
        chosen_traps: Set[int] = set()
        assignment: List[Tuple[int, int]] = []  # (gate_idx, trap_idx)

        for cost, gi, tj in edges:
            if gi in chosen_gates or tj in chosen_traps:
                continue
            # ensure trap tj can accept 2 ions for this gate this round
            if traps[tj].capacity >= 2:
                chosen_gates.add(gi); chosen_traps.add(tj)
                assignment.append((gi, tj))

        # 5) Build the new global arrangement.
        # First place matched gate pairs into their assigned traps.
        ionsPlaced: Set[int] = set()
        newArrangement = {T: [] for T in traps}

        for gi, tj in assignment:
            g = ready_MS[gi]
            a, d = g.ions
            newArrangement[traps[tj]].extend([a, d])
            ionsPlaced.update([a.idx, d.idx])

        # Then place all remaining ions, prefer staying put to minimize motion.
        for T in traps:
            for ion in list(T.ions):
                if ion.idx in ionsPlaced:
                    continue
                # try to keep in the same trap if capacity allows
                if len(newArrangement[T]) < T.capacity:
                    newArrangement[T].append(ion)
                    ionsPlaced.add(ion.idx)

        # Spill any still-unplaced ions into the nearest trap with room.
        if len(ionsPlaced) < len(arch.ions):
            for T in traps:
                for ion in list(T.ions):
                    if ion.idx in ionsPlaced:
                        continue
                    # find best trap with room
                    best = min(
                        (U for U in traps if len(newArrangement[U]) < U.capacity),
                        key=lambda U: move_cost(ion, U),
                    )
                    newArrangement[best].append(ion)
                    ionsPlaced.add(ion.idx)

        # 6) One global reconfiguration.
        reconfig = GlobalReconfigurations.physicalOperation(newArrangement, wiseArch)
        allOps.append(reconfig)
        reconfig.run()
        arch.refreshGraph()
        barriers.append(len(allOps))

        # 7) Run assigned two-qubit gates, one per trap, in parallel across traps.
        # (Here we just append them; your backend will schedule them concurrently.)
        for gi, tj in assignment:
            g = ready_MS[gi]
            trap = arch._manipulationTraps[tj]
            g.setTrap(trap)
            g.run()
            allOps.append(g)
            operationsLeft.remove(g)

        barriers.append(len(allOps))

    return allOps, barriers



from typing import Mapping, Sequence, Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
from typing import List, Sequence, Tuple




def ionRoutingWISEArch(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
    lookahead: int =4,
    subgridsize: Tuple[int, int, int] = (13,7, 1)
) -> Tuple[Sequence[Operation], Sequence[int]]:
    allOps: List[Operation] = []
    barriers=[]
    operationsLeft = list(operations)

    parallelismAllowed = wiseArch.m*wiseArch.n
    parallelPairs: List[List[Tuple[int, int]]]=[]
    toMoves: List[List[TwoQubitMSGate]] = []
    idx = 0
    _opstogothrough = list(operations).copy()
    while _opstogothrough:
        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()

        
            for op in _opstogothrough:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and len(op.ions)==1:
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(op.ions)
                
            for g in toRemove:
                _opstogothrough.remove(g)
            
            if len(toRemove)==0:
                break

        toRemove=[]
        ionsAdded = set()
        for op in _opstogothrough:
            if len(op.ions)==2 and len(toRemove)<parallelismAllowed:
                ion1, ion2 = op.ions
                ancilla, data = sorted(
                    (ion1, ion2), key=lambda ion: ion.label[0]=='D'
                )
                if (ancilla.idx in ionsAdded) or (data.idx in ionsAdded):
                    continue
                toRemove.append(op)
                if idx == len(parallelPairs):
                    parallelPairs.append([])
                parallelPairs[idx].append((ancilla.idx, data.idx))
                if idx == len(toMoves):
                    toMoves.append([])
                toMoves[idx].append(op)

                ionsAdded.add(ancilla.idx)
                ionsAdded.add(data.idx)
            else:
                ionsAdded.add(op.ions[0].idx)
      
        
        for g in toRemove:
            _opstogothrough.remove(g)


        idx+=int(len(toRemove)>0)


    idx=0
    oldArrangementArr = np.array([[0 for c in range(wiseArch.m*wiseArch.k)] for r in range(wiseArch.n)])
    newArrangementArr = np.array([[0 for c in range(wiseArch.m*wiseArch.k)] for r in range(wiseArch.n)])
    ionsSorted = sorted(list(arch.ions.values()), key=lambda ion: ion.pos[0]+100*ion.pos[1])
    for i, ion in enumerate(ionsSorted):
        oldArrangementArr[int(i/(wiseArch.m*wiseArch.k))][(i%(wiseArch.m*wiseArch.k))]=ion.idx

    
    # nsubgridsL = int(np.ceil((wiseArch.m*wiseArch.k)/subgridsize[0]))
    # nsubgridsD = int(np.ceil((wiseArch.n)/subgridsize[1]))
    P_arr = parallelPairs[:min(len(parallelPairs), lookahead)].copy()
    boundary_targets = [{} for _ in range(len(P_arr))]

    endcol = subgridsize[0]   # initial width of the growing slice
    endrow = subgridsize[1]   # initial height of the growing slice
    incrow = False            # zig-zag grow: col, row, col, row, ...

    while True :
        # 1) Build the current growing grid anchored at (0,0)
        currentGridList = []
        max_row = min(endrow, wiseArch.n)
        max_col = min(endcol, wiseArch.m * wiseArch.k)

        for r in range(max_row):
            if len(currentGridList) == r:
                currentGridList.append([])
            for c in range(max_col):
                currentGridList[r].append(oldArrangementArr[r][c])

        currentGrid = np.array(currentGridList)
        ionsInGrid = set(currentGrid.flatten())

        # Helper: frontier test for THIS slice
        def is_frontier_cell_local(d: int, c: int) -> bool:
            # using the same notion as inside _optimal_QMR_for_WISE
            return (d == max_row - 1) or (c == max_col - 1)

        # 2) Split pairs: which are inside this grid *for solving*?
        P_arr_in_grid = [[] for _ in range(len(P_arr))]
        # We will rebuild P_arr at the end of this iteration:
        new_P_arr = [[] for _ in range(len(P_arr))]

        for rn, arr in enumerate(P_arr):
            for (i1, i2) in arr:
                if i1 in ionsInGrid and i2 in ionsInGrid:
                    # This pair will be enforced in this SAT call
                    P_arr_in_grid[rn].append((i1, i2))
                else:
                    # Still not fully inside the current slice; keep for later iterations
                    new_P_arr[rn].append((i1, i2))

        # 3) Restrict boundary targets to ions that are in this grid (for all rounds)
        BT_in_grid = []
        for rn in range(len(P_arr)):
            bt_round = boundary_targets[rn]
            bt_slice = {
                ion: (d, c)
                for ion, (d, c) in bt_round.items()
                if ion in ionsInGrid
            }
            BT_in_grid.append(bt_slice)

        # 4) Solve on this slice: pairs = P_arr_in_grid, all BT_in_grid pins are passed in
        layouts_after = GlobalReconfigurations._optimal_QMR_for_WISE(
            currentGrid,
            P_arr_in_grid,
            k=wiseArch.k,
            BT=BT_in_grid,
        )

        # if sum(len(arr) for arr in new_P_arr)==0:
        #     break
            

        # 5) Collect the final positions of all ions in P_arr_in_grid (per round)
        #    so we can both:
        #    (a) update boundary_targets, and
        #    (b) decide which pairs are now fully interior and can be dropped.
        for rn, pairs_r in enumerate(P_arr_in_grid):
            if not pairs_r:
                # nothing new for this round in this slice
                continue

            layout_r = layouts_after[rn]
            # Build a small map: ion -> (row, col) in this slice
            pos = {}

            for rr in range(layout_r.shape[0]):
                for cc in range(layout_r.shape[1]):
                    ion = int(layout_r[rr, cc])
                    pos[ion] = (rr, cc)   # last assignment wins, but ions are unique anyway

            # 5a) Update boundary_targets for *all* ions in these pairs
            ions_in_pairs = set(np.array(pairs_r).flatten())
            for ion in ions_in_pairs:
                if ion in pos:
                    boundary_targets[rn][ion] = pos[ion]

            # 5b) Decide which pairs can be removed from P_arr
            for (i1, i2) in pairs_r:
                r1, c1 = pos[i1]
                r2, c2 = pos[i2]

                i1_frontier = is_frontier_cell_local(r1, c1)
                i2_frontier = is_frontier_cell_local(r2, c2)

                if i1_frontier or i2_frontier:
                    # at least one ion is still on the slice boundary
                    # -> its BT is still "soft-ish" (can be relaxed in later calls)
                    # -> keep this pair in P_arr so the constraint is re-enforced
                    new_P_arr[rn].append((i1, i2))
                else:
                    # both ions are interior in this slice:
                    #    their BT pins will be interior in *all future* larger slices
                    #    (because we always grow outward from (0,0)), so they effectively
                    #    become hard forever.
                    # -> we can safely drop this pair from P_arr.
                    pass

        # 6) Replace P_arr with the updated one, where:
        #    - pairs not yet fully inside the slice are kept
        #    - pairs that were inside but remained on the frontier are kept
        #    - only pairs that are now fully interior are removed
        P_arr = new_P_arr


        if endrow>wiseArch.n and endcol>wiseArch.m*wiseArch.k:
            break
        # 7) Grow the slice (zig-zag: first col, then row, then col, ...)
        if incrow:
            endrow+= subgridsize[2]
            incrow = endcol>wiseArch.m*wiseArch.k
        else:
            endcol+= subgridsize[2]
            incrow = endrow<=wiseArch.n
    # # Initial Ion Positioning
    newArrangement = {trap: [] for trap in arch._manipulationTraps}
    # layouts_after = GlobalReconfigurations._optimal_QMR_for_WISE(oldArrangementArr, P_arr, k=wiseArch.k)
    for d in range(wiseArch.n):
        for c in range(wiseArch.m*wiseArch.k):
            ionidx = layouts_after[0][d][c]
            newArrangementArr[d][c]=ionidx
            trap = arch.ions[oldArrangementArr[d][c]].parent
            newArrangement[trap].append(arch.ions[ionidx])

    reconfig  = GlobalReconfigurations.physicalOperation(newArrangement, wiseArch, oldArrangementArr, newArrangementArr)
    allOps.append(reconfig)
    reconfig.run()
    arch.refreshGraph()
    oldArrangementArr = newArrangementArr.copy()

    idx=0

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

        # # Determine new global configuration
        # ionsAdded = set()
        
        # toMoveCanDo: List[TwoQubitMSGate] = []

      
        # for op in toMove:
        #     ion1, ion2 = op.ions
        #     ancilla, data = sorted(
        #         (ion1, ion2), key=lambda ion: ion.label[0]=='D'
        #     )
        #     if (ancilla.idx in ionsAdded) or (data.idx in ionsAdded):
        #         continue
        #     currentTrap = data.parent
        #     if not isinstance(trap, ManipulationTrap):
        #         raise ValueError('Data Ion not in Trap!')
            
        #     # trapIn=min([trapIn for trapIn in newArrangement.keys() if trapIn.capacity>=len(newArrangement[trapIn])+2], key=lambda trapIn: (currentTrap.pos[0]-trapIn.pos[0])**2+(currentTrap.pos[1]-trapIn.pos[1])**2)

        #     # newArrangement[trapIn].append(ancilla)
        #     # newArrangement[trapIn].append(data)
        #     ionsAdded.add(ancilla.idx)
        #     ionsAdded.add(data.idx)
        #     toMoveCanDo.append(op)

    
        # for ion in arch.ions.values():
        #     if ion.idx not in ionsAdded:
        #         currentTrap = ion.parent
        #         #find trapIn such that trapIn.capacity<=len(newArrangement[trapIn])+1 that is closest to the current ion trap (could even be the current trap)
        #         trapIn=min([trapIn for trapIn in newArrangement.keys() if trapIn.capacity>=len(newArrangement[trapIn])+1], key=lambda trapIn: (currentTrap.pos[0]-trapIn.pos[0])**2+(currentTrap.pos[1]-trapIn.pos[1])**2)

        #         newArrangement[trapIn].append(ion)
        #         ionsAdded.add(ion.idx)

        # # determine the best new arrangement array 
        # _ionsToNewPos = {}
        # for trap, ions in newArrangement.items():
        #     for ion in ions:
        #         if ion.parent != trap:
        #             _ionsToNewPos[ion] = trap.pos
        #         else:
        #             _ionsToNewPos[ion] = ion.pos
        # ionsSorted = sorted(list(arch.ions.values()), key=lambda ion: _ionsToNewPos[ion][0]+100*_ionsToNewPos[ion][1])
        # for i, ion in enumerate(ionsSorted):
        #     newArrangementArr[int(i/(wiseArch.m*wiseArch.k))][(i%(wiseArch.m*wiseArch.k))]=ion.idx

        
       
    

        
        barriers.append(len(allOps))

        for op in toMoves[idx]:
            trap = op.getTrapForIons()
            op.setTrap(trap)
            op.run()
            allOps.append(op)
            operationsLeft.remove(op)

        barriers.append(len(allOps))
        idx+=1
        if idx>=len(toMoves):
            continue


        # layouts_after = GlobalReconfigurations._optimal_QMR_for_WISE(oldArrangementArr, parallelPairs[idx:min(len(parallelPairs), idx+lookahead)], k=wiseArch.k)
        # newArrangement = {trap: [] for trap in arch._manipulationTraps}
        # for d in range(wiseArch.n):
        #     for c in range(wiseArch.m*wiseArch.k):
        #         ionidx = layouts_after[0][d][c]
        #         newArrangementArr[d][c]=ionidx
        #         trap = arch.ions[oldArrangementArr[d][c]].parent
        #         newArrangement[trap].append(arch.ions[ionidx])

        # reconfig  = GlobalReconfigurations.physicalOperation(newArrangement, wiseArch, oldArrangementArr, newArrangementArr)
       
        # allOps.append(reconfig)
        # reconfig.run()
        # arch.refreshGraph()






        P_arr = parallelPairs[idx:min(len(parallelPairs), lookahead+idx)].copy()
        boundary_targets = [{} for _ in range(len(P_arr))]

        endcol = subgridsize[0]   # initial width of the growing slice
        endrow = subgridsize[1]   # initial height of the growing slice
        incrow = False            # zig-zag grow: col, row, col, row, ...

        while True:
            # 1) Build the current growing grid anchored at (0,0)
            currentGridList = []
            max_row = min(endrow, wiseArch.n)
            max_col = min(endcol, wiseArch.m * wiseArch.k)

            for r in range(max_row):
                if len(currentGridList) == r:
                    currentGridList.append([])
                for c in range(max_col):
                    currentGridList[r].append(oldArrangementArr[r][c])

            currentGrid = np.array(currentGridList)
            ionsInGrid = set(currentGrid.flatten())

            # Helper: frontier test for THIS slice
            def is_frontier_cell_local(d: int, c: int) -> bool:
                # using the same notion as inside _optimal_QMR_for_WISE
                return (d == max_row - 1) or (c == max_col - 1)

            # 2) Split pairs: which are inside this grid *for solving*?
            P_arr_in_grid = [[] for _ in range(len(P_arr))]
            # We will rebuild P_arr at the end of this iteration:
            new_P_arr = [[] for _ in range(len(P_arr))]

            for rn, arr in enumerate(P_arr):
                for (i1, i2) in arr:
                    if i1 in ionsInGrid and i2 in ionsInGrid:
                        # This pair will be enforced in this SAT call
                        P_arr_in_grid[rn].append((i1, i2))
                    else:
                        # Still not fully inside the current slice; keep for later iterations
                        new_P_arr[rn].append((i1, i2))

            # 3) Restrict boundary targets to ions that are in this grid (for all rounds)
            BT_in_grid = []
            for rn in range(len(P_arr)):
                bt_round = boundary_targets[rn]
                bt_slice = {
                    ion: (d, c)
                    for ion, (d, c) in bt_round.items()
                    if ion in ionsInGrid
                }
                BT_in_grid.append(bt_slice)

            # 4) Solve on this slice: pairs = P_arr_in_grid, all BT_in_grid pins are passed in
            layouts_after = GlobalReconfigurations._optimal_QMR_for_WISE(
                currentGrid,
                P_arr_in_grid,
                k=wiseArch.k,
                BT=BT_in_grid,
            )


            # 5) Collect the final positions of all ions in P_arr_in_grid (per round)
            #    so we can both:
            #    (a) update boundary_targets, and
            #    (b) decide which pairs are now fully interior and can be dropped.
            # doAddtoPArr=  sum(len(arr) for arr in new_P_arr) > 0
            for rn, pairs_r in enumerate(P_arr_in_grid):
                if not pairs_r:
                    # nothing new for this round in this slice
                    continue

                layout_r = layouts_after[rn]
                # Build a small map: ion -> (row, col) in this slice
                pos = {}

                for rr in range(layout_r.shape[0]):
                    for cc in range(layout_r.shape[1]):
                        ion = int(layout_r[rr, cc])
                        pos[ion] = (rr, cc)   # last assignment wins, but ions are unique anyway

                # 5a) Update boundary_targets for *all* ions in these pairs
                ions_in_pairs = set(np.array(pairs_r).flatten())
                for ion in ions_in_pairs:
                    if ion in pos:
                        boundary_targets[rn][ion] = pos[ion]

                # if not doAddtoPArr:
                #     continue
                # 5b) Decide which pairs can be removed from P_arr
                for (i1, i2) in pairs_r:
                    r1, c1 = pos[i1]
                    r2, c2 = pos[i2]

                    i1_frontier = is_frontier_cell_local(r1, c1)
                    i2_frontier = is_frontier_cell_local(r2, c2)

                    if i1_frontier or i2_frontier:
                        # at least one ion is still on the slice boundary
                        # -> its BT is still "soft-ish" (can be relaxed in later calls)
                        # -> keep this pair in P_arr so the constraint is re-enforced
                        new_P_arr[rn].append((i1, i2))
                    else:
                        # both ions are interior in this slice:
                        #    their BT pins will be interior in *all future* larger slices
                        #    (because we always grow outward from (0,0)), so they effectively
                        #    become hard forever.
                        # -> we can safely drop this pair from P_arr.
                        pass

            # 6) Replace P_arr with the updated one, where:
            #    - pairs not yet fully inside the slice are kept
            #    - pairs that were inside but remained on the frontier are kept
            #    - only pairs that are now fully interior are removed
            P_arr = new_P_arr

            if endrow>wiseArch.n and endcol>wiseArch.m*wiseArch.k:
                break

            # 7) Grow the slice (zig-zag: first col, then row, then col, ...)
            if incrow:
                endrow+= subgridsize[2]
                incrow = endcol>wiseArch.m*wiseArch.k
            else:
                endcol+= subgridsize[2]
                incrow = endrow<=wiseArch.n
        # # Initial Ion Positioning
        newArrangement = {trap: [] for trap in arch._manipulationTraps}
        # layouts_after = GlobalReconfigurations._optimal_QMR_for_WISE(oldArrangementArr, P_arr, k=wiseArch.k)
        for d in range(wiseArch.n):
            for c in range(wiseArch.m*wiseArch.k):
                ionidx = layouts_after[0][d][c]
                newArrangementArr[d][c]=ionidx
                trap = arch.ions[oldArrangementArr[d][c]].parent
                newArrangement[trap].append(arch.ions[ionidx])

        reconfig  = GlobalReconfigurations.physicalOperation(newArrangement, wiseArch, oldArrangementArr, newArrangementArr)
        allOps.append(reconfig)
        reconfig.run()
        arch.refreshGraph()




        oldArrangementArr = newArrangementArr.copy()


    return allOps, barriers
