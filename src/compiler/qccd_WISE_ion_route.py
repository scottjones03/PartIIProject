from typing import (
    Sequence,
    List,
    Tuple,
    Set,
    Dict,
    Mapping,
)
from collections import defaultdict

import numpy as np

from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler._qccd_WISE_ion_routing import *
from src.compiler.qccd_qubits_to_ions import *


def _grow_slice_and_route(
    oldArrangementArr: np.ndarray,
    wiseArch: QCCDWiseArch,
    P_arr: List[List[Tuple[int, int]]],
    subgridsize: Tuple[int, int, int],
    active_ions: List[int] = None
) -> List[np.ndarray]:
    """
    Internal helper: given a global arrangement (oldArrangementArr) and a small
    list of parallel ion pairs per round P_arr (length R):

      - Start with a small sub-grid (slice) anchored at (0, 0) of size
            (subgridsize[1] rows) × (subgridsize[0] columns).
      - Iteratively grow this slice by subgridsize[2] in either rows or columns
        (zig-zag: columns, then rows, etc.), but never beyond the physical
        dimension (wiseArch.n × wiseArch.m*wiseArch.k).
      - At each slice size:
          * restrict P_arr to those pairs whose ions are fully inside the slice,
          * restrict boundary targets BT to ions currently inside the slice,
          * call GlobalReconfigurations._optimal_QMR_for_WISE on this slice,
          * update per-round boundary_targets for ions involved in pairs,
          * drop pairs whose ions are now strictly interior (no longer on the
            slice frontier), to avoid re-enforcing them in future iterations.

      - Continue growing until the slice covers the full device (or the growth
        logic’s termination condition is met); return the layouts from the last
        SAT/MaxSAT call on this largest slice.

    This function is the **Level-1 slicer**: it keeps each SAT instance small by
    only ever giving _optimal_QMR_for_WISE a subgrid as input, and incrementally
    increasing that subgrid while “locking in” interior routing decisions.
    """
    # R = number of rounds we are solving in this call
    R = len(P_arr)
    if R == 0:
        return []

    boundary_targets: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(R)]

    endcol = subgridsize[0]   # initial width of the growing slice
    endrow = subgridsize[1]   # initial height of the growing slice
    step = subgridsize[2]     # increment step
    incrow = False            # zig-zag grow: col, row, col, row, ...

    layouts_after: List[np.ndarray] = []

    while True:
        # 1) Build the current growing grid anchored at (0,0)
        currentGridList: List[List[int]] = []
        max_row = min(endrow, wiseArch.n)
        max_col = min(endcol, wiseArch.m * wiseArch.k)
        isGridFull = 1-max_row/wiseArch.n ,1-( max_col/(wiseArch.m*wiseArch.k))

        for r in range(max_row):
            if len(currentGridList) == r:
                currentGridList.append([])
            for c in range(max_col):
                currentGridList[r].append(oldArrangementArr[r][c])

        currentGrid = np.array(currentGridList, dtype=int)
        ionsInGrid = set(currentGrid.flatten())

        # Helper: frontier test for THIS slice
        def is_frontier_cell_local(d: int, c: int) -> bool:
            return (d == max_row - 1) or (c == max_col - 1)

        # 2) Split pairs into inside-this-grid vs still-outside
        P_arr_in_grid: List[List[Tuple[int, int]]] = [[] for _ in range(R)]
        new_P_arr: List[List[Tuple[int, int]]] = [[] for _ in range(R)]

        for rn, arr in enumerate(P_arr):
            for (i1, i2) in arr:
                if (i1 in ionsInGrid) and (i2 in ionsInGrid):
                    P_arr_in_grid[rn].append((i1, i2))   # enforced in this call
                else:
                    new_P_arr[rn].append((i1, i2))       # wait for later

        # 3) Restrict boundary targets to ions that are in this subgrid
        BT_in_grid: List[Dict[int, Tuple[int, int]]] = []
        for rn in range(R):
            bt_round = boundary_targets[rn]
            bt_slice = {
                ion: (d, c)
                for ion, (d, c) in bt_round.items()
                if ion in ionsInGrid
            }
            BT_in_grid.append(bt_slice)

        

   
        # 4) Solve this slice:
        #    Level-2 and Level-3 happen inside _optimal_QMR_for_WISE:
        #      - Level-2: SAT + D-minimisation,
        #      - Level-3: small MaxSAT to avoid boundary cells.
        layouts_after = GlobalReconfigurations._optimal_QMR_for_WISE(
            currentGrid,
            P_arr_in_grid,
            k=wiseArch.k,
            BT=BT_in_grid,
            active_ions=active_ions,
            wB_col=isGridFull[1],
            wB_row=isGridFull[0]
        )

        # 5) Update boundary_targets and decide which pairs to drop
        for rn, pairs_r in enumerate(P_arr_in_grid):
            if not pairs_r:
                continue

            layout_r = layouts_after[rn]
            pos: Dict[int, Tuple[int, int]] = {}

            for rr in range(layout_r.shape[0]):
                for cc in range(layout_r.shape[1]):
                    ion = int(layout_r[rr, cc])
                    pos[ion] = (rr, cc)

            ions_in_pairs = set(np.array(pairs_r).flatten())
            for ion in ions_in_pairs:
                if ion in pos:
                    boundary_targets[rn][ion] = pos[ion]

            # Decide which pairs can be removed (both ions interior)
            for (i1, i2) in pairs_r:
                r1, c1 = pos[i1]
                r2, c2 = pos[i2]

                i1_frontier = is_frontier_cell_local(r1, c1)
                i2_frontier = is_frontier_cell_local(r2, c2)

                if i1_frontier or i2_frontier:
                    # still touching the slice boundary → keep this pair in P_arr
                    new_P_arr[rn].append((i1, i2))
                # else: drop pair forever; they are interior and stay hard in future slices

        # 6) Update P_arr for the next growth step
        P_arr = new_P_arr

        # 7) Termination: slice has fully covered the device
        if (endrow >= wiseArch.n) and (endcol >= wiseArch.m * wiseArch.k):
            break

        # 8) Grow the slice (zig-zag: first columns, then rows, ...)
        if incrow:
            endrow += step
            incrow = endcol >= wiseArch.m * wiseArch.k
        else:
            endcol += step
            incrow = endrow <= wiseArch.n

    return layouts_after


def _apply_layout_as_reconfiguration(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    oldArrangementArr: np.ndarray,
    newArrangementArr: np.ndarray,
    layouts_after: List[np.ndarray],
    allOps: List[Operation],
) -> np.ndarray:
    """
    Internal helper: take the first layout in layouts_after (round 0 layout of
    the slice solver) and:

      - interpret it as a new global arrangement of ions over the full
        wiseArch.n × (wiseArch.m*wiseArch.k) grid,
      - group ions per manipulation trap,
      - build a GlobalReconfigurations.physicalOperation that moves ions from
        oldArrangementArr to newArrangementArr,
      - run that operation and append it to allOps,
      - refresh the architecture graph,
      - return the updated oldArrangementArr to be used as the new baseline.

    This encapsulates the “apply one big subgrid-based reconfiguration step”
    that is duplicated in the original implementation.
    """
    newArrangement: Dict[ManipulationTrap, List[Ion]] = {
        trap: [] for trap in arch._manipulationTraps
    }

    for d in range(wiseArch.n):
        for c in range(wiseArch.m * wiseArch.k):
            ionidx = int(layouts_after[0][d][c])
            newArrangementArr[d][c] = ionidx
            # Note: trap membership is based on the *old* arrangement
            trap = arch.ions[int(oldArrangementArr[d][c])].parent
            newArrangement[trap].append(arch.ions[ionidx])

    reconfig = GlobalReconfigurations.physicalOperation(
        newArrangement, wiseArch, oldArrangementArr, newArrangementArr
    )
    allOps.append(reconfig)
    reconfig.run()
    arch.refreshGraph()

    return newArrangementArr.copy()


def ionRoutingWISEArch(
    arch: QCCDArch,
    wiseArch: QCCDWiseArch,
    operations: Sequence[QubitOperation],
    lookahead: int = 2,
    subgridsize: Tuple[int, int, int] = (6, 4, 1),
) -> Tuple[Sequence[Operation], Sequence[int]]:
    """
    Route a WISE-style QCCD architecture in **three optimisation levels**:

      Level 1 – Spatial slicing and incremental subgrid growth
      --------------------------------------------------------
      - Given a global ion layout and a sequence of two-qubit rounds, we:
          1. Encode the grid as an n×(m·k) ion index array oldArrangementArr.
          2. Partition the two-qubit MS gates into maximally parallel rounds
             (parallelPairs, toMoves).
          3. For each window of rounds P_arr (of length ≤ lookahead):
                * Start with a small subgrid anchored at (0,0) of size
                      subgridsize[1] rows × subgridsize[0] columns.
                * Iteratively **grow** this subgrid (by subgridsize[2] rows or
                  columns in a zig-zag fashion).
                * At each growth step, restrict:
                    - ion pairs P_arr to those fully inside the subgrid,
                    - boundary targets BT to ions currently in the subgrid.
                * Call the per-slice optimiser (Levels 2 & 3) on this subgrid.
                * Use the resulting layout to:
                    - update per-round boundary targets of ions,
                    - drop pairs whose ions have become strictly interior.
                * Once the subgrid covers the full device, apply the final slice
                  layout as a single physical reconfiguration.

      Level 2 – Per-slice D-minimising SAT (inside _optimal_QMR_for_WISE)
      --------------------------------------------------------------------
      - For a given slice (currentGrid) and a small number of rounds P_arr[r],
        _optimal_QMR_for_WISE first:
          * builds a purely hard CNF encoding:
              - exact-one-per-cell layouts a[r],
              - per-ion row/column targets t[r], x[r],
              - block membership w[r],
              - pair constraints (same row & block),
              - BT pins for reserved ions,
              - layout consistency a[r+1] <-> (x[r], t[r]),
              - row presence p and y[r,k,c,i] with reserved-aware semantics,
              - a movement bound: max horizontal/vertical displacement ≤ D.
          * performs a **binary search on D**, solving SAT instances until it
            finds the smallest D* for which the CNF is satisfiable.

        This yields a per-slice routing that respects BT and pairs while
        minimising the maximum displacement D* within that slice.

      Level 3 – Per-slice boundary-aware MaxSAT (also inside _optimal_QMR_for_WISE)
      ----------------------------------------------------------------------------
      - With D fixed to D*, the same structural CNF is rebuilt as a WCNF,
        and small soft clauses are added to discourage interacting ions from
        landing on the outermost row / column of the slice:
            ¬x[r, i, last_column],  ¬t[r, i, last_row]
      - A MaxSAT solver (RC2) is then used to minimise the weighted number
        of violated soft clauses, without breaking any of the hard constraints
        or increasing D beyond D*.

    The overall behaviour of ionRoutingWISEArch is thus:

      1. Partition the circuit’s two-qubit gates into parallel rounds (by ions).
      2. For each “chunk” of up to `lookahead` rounds:
           - Run the Level-1 slicer, which repeatedly calls the Level-2/3
             subgrid optimiser until the entire device is covered.
           - Promote the final subgrid layout to a global physical
             reconfiguration operation, append it to allOps, and update the
             global ion positions.
      3. Between these reconfigurations, execute all single-qubit operations
         that fit without further routing, and then the scheduled two-qubit
         MS gates for that chunk, inserting barriers to preserve the time
         structure for later analysis.

    Returns:
        allOps    : the full, time-ordered list of physical operations,
                    including reconfigurations, single-qubit gates, and MS gates.
        barriers  : indices in allOps that act as “barriers” between logical
                    layers / routing phases (useful for parallelisation analysis).
    """
    allOps: List[Operation] = []
    barriers: List[int] = []
    operationsLeft = list(operations)

    # ------------------------------------------------------------------
    # 1) Build parallel rounds of two-qubit MS gates (parallelPairs/toMoves)
    # ------------------------------------------------------------------
    parallelismAllowed = wiseArch.m * wiseArch.n
    parallelPairs: List[List[Tuple[int, int]]] = []
    toMoves: List[List[TwoQubitMSGate]] = []

    idx = 0
    _opstogothrough = list(operations).copy()

    while _opstogothrough:
        # First, greedily take as many disjoint 1-qubit ops as possible
        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()

            for op in _opstogothrough:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and len(op.ions) == 1:
                    toRemove.append(op)
                ionsInvolved = ionsInvolved.union(op.ions)

            for g in toRemove:
                _opstogothrough.remove(g)

            if len(toRemove) == 0:
                break

        # Then, form one round of disjoint 2-qubit MS gates
        toRemove = []
        ionsAdded: Set[int] = set()
        for op in _opstogothrough:
            if len(op.ions) == 2 and len(toRemove) < parallelismAllowed:
                ion1, ion2 = op.ions
                ancilla, data = sorted(
                    (ion1, ion2), key=lambda ion: ion.label[0] == "D"
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

        idx += int(len(toRemove) > 0)

    # ------------------------------------------------------------------
    # 2) Encode initial ion positions into oldArrangementArr
    # ------------------------------------------------------------------
    oldArrangementArr = np.array(
        [[0 for _ in range(wiseArch.m * wiseArch.k)] for _ in range(wiseArch.n)],
        dtype=int,
    )
    newArrangementArr = np.array(
        [[0 for _ in range(wiseArch.m * wiseArch.k)] for _ in range(wiseArch.n)],
        dtype=int,
    )

    ionsSorted = sorted(
        list(arch.ions.values()), key=lambda ion: ion.pos[0] + 100 * ion.pos[1]
    )
    for i, ion in enumerate(ionsSorted):
        r = i // (wiseArch.m * wiseArch.k)
        c = i % (wiseArch.m * wiseArch.k)
        oldArrangementArr[r][c] = ion.idx

    active_ions = [ion.idx for ion in ionsSorted if not isinstance(ion, SpectatorIon)]

    # ------------------------------------------------------------------
    # 3) Initial global reconfiguration via Level-1/2/3 on the first chunk
    # ------------------------------------------------------------------
    P_arr = parallelPairs[: min(len(parallelPairs), lookahead)].copy()
    layouts_after = _grow_slice_and_route(
        oldArrangementArr, wiseArch, P_arr, subgridsize, active_ions=active_ions
    )
    oldArrangementArr = _apply_layout_as_reconfiguration(
        arch, wiseArch, oldArrangementArr, newArrangementArr, layouts_after, allOps
    )

    idx = 0

    # ------------------------------------------------------------------
    # 4) Execute operations, routing between parallel MS rounds as needed
    # ------------------------------------------------------------------
    while operationsLeft:
        # 4a) Run as many single-qubit operations as possible without routing
        while True:
            toRemove: List[Operation] = []
            ionsInvolved: Set[Ion] = set()
            for op in operationsLeft:
                trap = op.getTrapForIons()
                if ionsInvolved.isdisjoint(op.ions) and trap and len(op.ions) == 1:
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

        # 4b) Execute one parallel round of two-qubit MS gates
        barriers.append(len(allOps))
        for op in toMoves[idx]:
            trap = op.getTrapForIons()
            op.setTrap(trap)
            op.run()
            allOps.append(op)
            operationsLeft.remove(op)

        barriers.append(len(allOps))
        idx += 1
        if idx >= len(toMoves):
            # no more MS rounds → loop will exit after remaining 1q ops
            continue

        # 4c) Between MS rounds: re-route using next lookahead window of pairs
        P_arr = parallelPairs[idx : min(len(parallelPairs), lookahead + idx)].copy()
        layouts_after = _grow_slice_and_route(
            oldArrangementArr, wiseArch, P_arr, subgridsize, active_ions=active_ions
        )
        oldArrangementArr = _apply_layout_as_reconfiguration(
            arch, wiseArch, oldArrangementArr, newArrangementArr, layouts_after, allOps
        )

    return allOps, barriers