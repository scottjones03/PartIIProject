import numpy as np
from typing import (
    Sequence,
    List,
    Optional,
    Callable,
    Any,
    Mapping
)
import abc
from src.utils.qccd_nodes import *
from typing import List, Sequence, Dict, Tuple
from collections import defaultdict, deque
from collections import defaultdict
from dataclasses import dataclass, field
import heapq, math, bisect
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from pysat.formula import IDPool, WCNF, CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22  
from pysat.examples.rc2 import RC2
import time

class Operation:
    KEY: Operations

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        self._run = run
        self._kwargs = dict(kwargs)
        self._involvedIonsForLabel: List[Ion] = []
        self._involvedComponents: List[QCCDComponent] = involvedComponents
        self._addOns = ""
        self._fidelity: float = 1.0
        self._dephasingFidelity: float = 1.0
        self._operationTime: float = 0.0

    def addComponent(self, component: QCCDComponent) -> None:
        self._involvedComponents.append(component)

    @property
    def involvedComponents(self) -> Sequence[QCCDComponent]:
        return self._involvedComponents

    @property
    def color(self) -> str:
        return "lightgreen"

    @property
    def involvedIonsForLabel(self) -> Sequence[Ion]:
        return self._involvedIonsForLabel

    @property
    def label(self) -> str:
        return self.KEY.name + self._addOns

    @property
    @abc.abstractmethod
    def isApplicable(self) -> bool:
        return all(self.KEY in component.allowedOperations for component in self.involvedComponents)
    
    @abc.abstractmethod
    def _checkApplicability(self) -> None:
        for component in self.involvedComponents:
            if self.KEY not in component.allowedOperations:
                raise ValueError(f"Component {component} with index {component.idx} cannot complete {self.KEY.name}")

    @classmethod
    @abc.abstractmethod
    def physicalOperation(cls) -> "Operation": ...

    @abc.abstractmethod
    def calculateFidelity(self) -> None: ...

    @abc.abstractmethod
    def calculateDephasingFidelity(self) -> None: ...

    @abc.abstractmethod
    def calculateOperationTime(self) -> None: ...

    @abc.abstractmethod
    def _generateLabelAddOns(self) -> None: ...

    def run(self) -> None:
        self._checkApplicability()
        self.calculateOperationTime()
        self.calculateFidelity()
        self.calculateDephasingFidelity()
        self._run(())
        self._generateLabelAddOns()

    def dephasingFidelity(self) -> float:
        # Deprecated!
        return self._dephasingFidelity

    def fidelity(self) -> float:
        return self._fidelity
    
    def operationTime(self) -> float:
        return self._operationTime


class CrystalOperation(Operation):
    T2 = 2.2 # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents, **kwargs)
        self._trap: Trap = kwargs["trap"]


    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/self.T2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    @property
    def ionsInfluenced(self) -> Sequence[Ion]:
        return self._trap.ions
    
    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = list(self._trap.ions)
        self._addOns = ""
        for ion in self._involvedIonsForLabel:
            self._addOns += f" {ion.label}"


class GlobalReconfigurations(Operation):
    KEY = Operations.GLOBAL_RECONFIG

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._wiseArch: QCCDWiseArch = kwargs['wiseArch']
        self._reconfigTime: float = kwargs['reconfigTime']

    def calculateOperationTime(self) -> None:
        self._operationTime =  self._reconfigTime

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        # FIXME might be inaccurate
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/2.2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330


    def _generateLabelAddOns(self) -> None:
        self._addOns = f""

    @property
    def isApplicable(self) -> bool:
        return True
    
    def _checkApplicability(self) -> None:
        return True

    @classmethod
    def physicalOperation(
        cls, arrangement: Mapping[Trap, Sequence[Ion]], wiseArch: QCCDWiseArch, oldAssignment: Sequence[Sequence[int]], newAssignment: Sequence[Sequence[int]],
    ):
        heatingRates, reconfigTime = cls._runOddEvenReconfig(wiseArch, arrangement, oldAssignment, newAssignment)
        def run():
            for trap in arrangement.keys():
                while trap.ions:
                    trap.removeIon(trap.ions[0])
            for trap, ions in arrangement.items():
                for i, ion in enumerate(ions):
                    trap.addIon(ion, offset=i)
                    ion.addMotionalEnergy(heatingRates[ion.idx])
        return cls(
            run=lambda _: run(),
            involvedComponents=list(arrangement.keys()),
            wiseArch=wiseArch,
            reconfigTime=reconfigTime

        )

    @staticmethod
    def _optimal_phaseB_min_passes(A_in: np.ndarray, T_in: np.ndarray) -> Tuple[int, np.ndarray]:
        A = np.asarray(A_in, dtype=int)
        T = np.asarray(T_in, dtype=int)
        n, m = A.shape
        INF = 10**9

        # ---- dest row for every ion id ----
        ion_to_dest: Dict[int, int] = {}
        for rr in range(n):
            for cc in range(m):
                ion_to_dest[int(T[rr, cc])] = rr

        # ---- Remaining positions in each row grouped by dest ----
        def build_remaining_pos_by_dest(mat):
            buckets = []
            for r in range(n):
                dmap = defaultdict(list)
                for j in range(m):
                    ion = int(mat[r, j])
                    dmap[ion_to_dest[ion]].append(j)
                for d in dmap:
                    dmap[d].sort()
                buckets.append(dmap)
            return buckets

        # === FAST EXACT per-row bottleneck distance on a line (no matching needed) ===
        # Given remaining source positions per row (flattened multiset of j's)
        # and the set of remaining global columns, the minimal possible max |j-c|
        # equals max_i |sorted(J)[i] - sorted(C)[i]|. (Monge / 1D bottleneck assignment)
        def row_bottleneck_distance_sorted(Js_sorted: List[int], cols_sorted: List[int]) -> int:
            if not Js_sorted:
                return 0
            # lengths must match for feasibility at the row level
            if len(Js_sorted) != len(cols_sorted):
                return INF
            return max(abs(j - c) for j, c in zip(Js_sorted, cols_sorted))

        # ---- exact leaf cost of a mapping ----
        def exact_pass_cost(desired: np.ndarray) -> int:
            total = 0
            invpos = []
            for r in range(n):
                mp = {}
                for j in range(m):
                    mp[int(A[r, j])] = j
                invpos.append(mp)
            for r in range(n):
                row_max = 0
                for c in range(m):
                    ion = int(desired[r, c])
                    j0  = invpos[r][ion]
                    row_max = max(row_max, abs(j0 - c))
                total = max(total, row_max)
            return total

        # ---- Priority-queue node ----
        @dataclass(order=True)
        class Node:
            priority: int
            c_idx: int = field(compare=False)               # count of columns already assigned
            desired_partial: np.ndarray = field(compare=False)  # (n x m), -1 where unfilled
            Ar_committed: List[int] = field(compare=False)  # per-row committed max displacement (A_r)
            remaining_by_dest: List[Dict[int, List[int]]] = field(compare=False)  # per row, dest -> sorted js
            remaining_cols: List[int] = field(compare=False)  # remaining column indices (unordered)

        # ---- initial state ----
        remaining0 = build_remaining_pos_by_dest(A)
        desired0 = -np.ones((n, m), dtype=int)
        Ar0 = [0] * n
        remaining_cols0 = list(range(m))

        # === admissible LB using fast bottleneck ===
        # LB = max_r max(Ar[r], Br), with Br computed by the closed-form formula
        def admissible_lb(Ar: List[int], remaining_by_dest: List[Dict[int, List[int]]], rem_cols: List[int]) -> int:
            if not rem_cols:
                return max(Ar) if Ar else 0
            cols_sorted = sorted(rem_cols)
            lb = 0
            for r in range(n):
                # flatten remaining j's for this row
                Js = []
                for lst in remaining_by_dest[r].values():
                    Js.extend(lst)
                Js.sort()
                Br = row_bottleneck_distance_sorted(Js, cols_sorted)
                lr = max(Ar[r], Br)
                if lr > lb:
                    lb = lr
            return lb

        # === pick next column (MRV-style): choose the most constrained column ===
        # Score each remaining c by the sum over rows of their *local* cheapest displacement to c.
        def choose_next_column(rem_cols: List[int], remaining_by_dest: List[Dict[int, List[int]]]) -> int:
            best_c, best_score = None, INF
            for c in rem_cols:
                score = 0
                feasible = True
                for r in range(n):
                    # minimal achievable |j-c| for any dest available in this row
                    best = INF
                    for lst in remaining_by_dest[r].values():
                        if not lst: 
                            continue
                        i = bisect.bisect_left(lst, c)
                        if i < len(lst): best = min(best, abs(lst[i] - c))
                        if i > 0:        best = min(best, abs(lst[i-1] - c))
                    if best == INF:
                        feasible = False
                        break
                    score += best
                    if score >= best_score:  # cheap cutoff
                        break
                if feasible and score < best_score:
                    best_score = score
                    best_c = c
            # Fallback (shouldn’t happen): return first remaining
            return best_c if best_c is not None else rem_cols[0]

        # ---- enumerate feasible column matchings (rows -> dests) with per-row availability ----
        # We explore in increasing "per-row nearest displacement" threshold for this column.
        def matchings_for_column(c: int, remaining_by_dest: List[Dict[int, List[int]]]):
            # Build n x n matrix of per-row minimal |j-c| if that dest is chosen; INF if unavailable.
            cost = np.full((n, n), INF, dtype=int)
            argj = [[None]*n for _ in range(n)]
            for r in range(n):
                for d, pos_list in remaining_by_dest[r].items():
                    if pos_list:
                        i = bisect.bisect_left(pos_list, c)
                        cand = []
                        if i < len(pos_list): cand.append(pos_list[i])
                        if i > 0:            cand.append(pos_list[i-1])
                        if cand:
                            bestj = min(cand, key=lambda jp: abs(jp - c))
                            cost[r, d] = abs(bestj - c)
                            argj[r][d] = bestj
            finite = cost[cost < INF]
            if finite.size == 0:
                return
            thresholds = sorted(set(int(x) for x in finite))
            # Use Hungarian on the masked matrix for each increasing threshold
            from scipy.optimize import linear_sum_assignment
            BIG = 10**7
            for t in thresholds:
                C = np.where(cost <= t, cost, BIG)
                rows, cols = linear_sum_assignment(C)
                if len(rows) == n and all(C[r, d] < BIG for r, d in zip(rows, cols)):
                    perm_d = np.zeros(n, dtype=int)
                    chosen_j = [None]*n
                    for r, d in zip(rows, cols):
                        perm_d[r] = d
                        chosen_j[r] = argj[r][d]
                    yield t, perm_d, chosen_j

        # ---- Branch & Bound (best-first on LB) ----
        best_desired: Optional[np.ndarray] = None
        best_cost: int = math.inf

        root_lb = admissible_lb(Ar0, remaining0, remaining_cols0)
        pq = [Node(priority=root_lb, c_idx=0, desired_partial=desired0, Ar_committed=Ar0,
                   remaining_by_dest=remaining0, remaining_cols=remaining_cols0)]

        while pq:
            node = heapq.heappop(pq)
            lb_here = node.priority
            if lb_here >= best_cost:
                continue
            if node.c_idx == m:
                cost_leaf = exact_pass_cost(node.desired_partial)
                if cost_leaf < best_cost:
                    best_cost = cost_leaf
                    best_desired = node.desired_partial.copy()
                continue

            # === choose the most constrained column to branch on ===
            c = choose_next_column(node.remaining_cols, node.remaining_by_dest)

            # Try feasible column matchings in increasing local bottleneck for this column
            for _, perm_d, chosen_j in matchings_for_column(c, node.remaining_by_dest):
                desired_next = node.desired_partial.copy()
                Ar_next = node.Ar_committed[:]
                rem_cols_next = [cc for cc in node.remaining_cols if cc != c]

                rem_next = []
                feasible = True
                for r in range(n):
                    dd = {d: lst.copy() for d, lst in node.remaining_by_dest[r].items()}
                    rem_next.append(dd)
                    d = int(perm_d[r])
                    j_sel = chosen_j[r]
                    if j_sel is None or j_sel not in rem_next[r].get(d, []):
                        feasible = False
                        break
                    rem_next[r][d].remove(j_sel)
                    ion = int(A[r, j_sel])
                    desired_next[r, c] = ion
                    Ar_next[r] = max(Ar_next[r], abs(j_sel - c))
                if not feasible:
                    continue

                lb_child = admissible_lb(Ar_next, rem_next, rem_cols_next)
                if lb_child >= best_cost:
                    continue
                heapq.heappush(pq, Node(priority=lb_child, c_idx=node.c_idx+1,
                                        desired_partial=desired_next, Ar_committed=Ar_next,
                                        remaining_by_dest=rem_next, remaining_cols=rem_cols_next))

        if best_desired is None:
            raise RuntimeError("No feasible Phase-B assignment found (check inputs).")
        return best_cost, best_desired

    @staticmethod
    def sanity(A, T):
        A = np.asarray(A); T = np.asarray(T)
        n, m = A.shape

        # sets of IDs
        S_A = set(int(x) for x in A.ravel())
        S_T = set(int(x) for x in T.ravel())
        assert S_A == S_T, "Ion ID sets differ between A and T"

        # dest map
        ion_to_dest = {}
        for r in range(n):
            for c in range(m):
                ion_to_dest[int(T[r,c])] = r

        # per-destination totals (should be m each)
        from collections import Counter
        cnt_dest = Counter(ion_to_dest[int(x)] for x in A.ravel())
        for d in range(n):
            assert cnt_dest[d] == m, f"Destination {d} occurs {cnt_dest[d]} times in A (expected m={m})"

        # per-row totals by dest (for curiosity; they should sum to m)
        row_counts = []
        for r in range(n):
            cR = Counter(ion_to_dest[int(A[r,j])] for j in range(m))
            assert sum(cR.values()) == m, f"Row {r} has {sum(cR.values())} ions (expected m={m})"
            row_counts.append(cR)
        return ion_to_dest, row_counts
    

    @staticmethod
    def _optimal_phaseB_min_passes_sat_heavy(A_in, T_in):
        """
        Exact Phase-B optimizer (WISE) via SAT + binary search on P (layers).
        Minimizes the true makespan in odd-even transposition: the minimum number of
        odd/even layers needed so that:
        (i) rows evolve only by adjacent swaps compatible with layer parity, and
        (ii) at final layer P, in each global column c, every destination row d
            appears at most once (so exactly once if feasible).
        Returns:
            P_opt (int), desired (np.ndarray[int] of shape [n, m])
        """
        import numpy as np
        from pysat.formula import CNF, IDPool
        from pysat.card import CardEnc, EncType
        from pysat.solvers import Minisat22

        A = np.asarray(A_in, dtype=int)
        T = np.asarray(T_in, dtype=int)
        n, m = A.shape

        # Map ion id -> destination row (from T)
        ion_to_dest = {}
        for rr in range(n):
            for cc in range(m):
                ion_to_dest[int(T[rr, cc])] = rr

        # Helper: build and solve SAT for a given number of layers P
        def solve_P(P: int):
            vpool = IDPool(start_from=1)
            cnf = CNF()

            # Tokens are indexed by their ORIGINAL column j0 in their row.
            # y(r,i,t,j0): in row r, at time t, position i holds token j0 (the ion A[r, j0]).
            def Y(r, i, t, j0): return vpool.id(('y', r, i, t, j0))
            # s(r,i,t): in row r, layer t, the pair (i,i+1) swaps (only if that pair is active at layer t)
            def S(r, i, t):     return vpool.id(('s', r, i, t))

            # ---- Initial layer t = 0: each position i holds exactly token j0=i ----
            for r in range(n):
                # Exactly-one at (r,i,0) over j0 = 0..m-1, with the unique true being j0 = i
                for i in range(m):
                    # Force Y(r,i,0,i) = True, and Y(r,i,0,j0!=i) = False
                    cnf.append([ Y(r, i, 0, i) ])
                    for j0 in range(m):
                        if j0 != i:
                            cnf.append([ -Y(r, i, 0, j0) ])
                # Also: per token j0 at t=0 is at exactly one position i (namely i=j0)
                for j0 in range(m):
                    cnf.append([ Y(r, j0, 0, j0) ])  # unit, already set

            # ---- For each layer t = 0..P-1, encode odd/even pair transitions ----
            for t in range(P):
                # Layer parity: odd-even transposition network
                # t=0: "odd" pairs (0,1),(2,3),...   (i even)
                # t=1: "even" pairs (1,2),(3,4),...  (i odd)
                start = 0 if (t % 2 == 0) else 1
                active_pairs = set(range(start, m-1, 2))
                inactive_positions = set(range(m))  # will remove endpoints covered by active pairs
                for i in active_pairs:
                    inactive_positions.discard(i)
                    inactive_positions.discard(i+1)

                for r in range(n):
                    row_tokens = range(m)  # tokens 0..m-1 for this row

                    # For every active adjacent pair (i,i+1), choose swap vs stay and propagate tokens accordingly.
                    for i in active_pairs:
                        sv = S(r, i, t)
                        for j0 in row_tokens:
                            # sv => (Y[r,i,t,j0] -> Y[r,i+1,t+1,j0]) AND (Y[r,i+1,t,j0] -> Y[r,i,t+1,j0])
                            cnf.append([ -sv, -Y(r,i,t,j0),   Y(r,i+1,t+1,j0) ])
                            cnf.append([ -sv, -Y(r,i+1,t,j0), Y(r,i,  t+1,j0) ])
                            # ~sv => (Y[r,i,t,j0] -> Y[r,i,  t+1,j0]) AND (Y[r,i+1,t,j0] -> Y[r,i+1,t+1,j0])
                            cnf.append([  sv, -Y(r,i,t,j0),   Y(r,i,  t+1,j0) ])
                            cnf.append([  sv, -Y(r,i+1,t,j0), Y(r,i+1,t+1,j0) ])

                    # For positions NOT in any active pair at layer t: they must carry over unchanged to t+1.
                    for i in inactive_positions:
                        for j0 in row_tokens:
                            cnf.append([ -Y(r,i,t,j0), Y(r,i,t+1,j0) ])  # Y(r,i,t,j0) -> Y(r,i,t+1,j0)

                    # Exactly-one constraints at time t+1:
                    # (a) For each position i, exactly one token j0 occupies it.
                    for i in range(m):
                        lits = [Y(r,i,t+1,j0) for j0 in row_tokens]
                        cnf.extend(CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool))
                    # (b) For each token j0, it occupies exactly one position i.
                    for j0 in row_tokens:
                        lits = [Y(r,i,t+1,j0) for i in range(m)]
                        cnf.extend(CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool))

            # ---- Final WISE column uniqueness at time P ----
            # For each global column c and destination row d, at-most-one across all rows r of ions with dest d placed at (r,c,P).
            for c in range(m):
                for d in range(n):
                    lits = []
                    for r in range(n):
                        for j0 in range(m):
                            ion = int(A[r, j0])               # ion id of token j0 in row r
                            if ion_to_dest[ion] == d:
                                lits.append( Y(r, c, P, j0) )  # if token j0 ends at col c at time P
                    # If no token has dest d for this column, it's fine (no clause).
                    if lits:
                        cnf.extend(CardEnc.atmost(lits=lits, bound=1, encoding=EncType.pairwise, vpool=vpool))

            # Solve
            with Minisat22(bootstrap_with=cnf.clauses) as sat:
                ok = sat.solve()
                if not ok:
                    return False, None
                model = set(l for l in sat.get_model() if l > 0)

            # Extract desired layout from y at time P
            desired = -np.ones((n, m), dtype=int)
            for r in range(n):
                for c in range(m):
                    placed = False
                    for j0 in range(m):
                        if vpool.id(('y', r, c, P, j0)) in model:
                            desired[r, c] = int(A[r, j0])
                            placed = True
                            break
                    if not placed:
                        # Defensive: should not happen because of equals constraints
                        return False, None
            return True, desired

        # Binary-search the minimal number of layers P
        lo, hi = 0, m  # hi can be m; worst case m layers for odd-even transposition
        best_P, best_desired = None, None
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, des = solve_P(mid)
            if ok:
                best_P, best_desired = mid, des
                hi = mid - 1
            else:
                lo = mid + 1

        if best_desired is None:
            raise RuntimeError("No feasible Phase-B assignment found with time-expanded odd/even encoding.")
        return best_P, best_desired


    @staticmethod
    def _optimal_QMR_for_WISE(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,    # unused in D-min version
        wV: List[int] = None,    # unused in D-min version
        wB_col: int = 1,
        wB_row: int = 1
    ) -> List[np.ndarray]:
        """
        Two-level optimizer for WISE:

        Level 1 (SAT + binary search):
            Minimize D = max(horizontal displacement, vertical displacement)
            subject to:
            - BT pins (hard),
            - pair constraints (same dest row, same block),
            - exact-one-per-cell layouts in each round,
            - no two ions from the same *current row* target the same column.

        Level 2 (MaxSAT at D*):
            With D fixed to D*, re-solve with the same hard constraints
            and add soft clauses to discourage interacting ions
            from ending on boundary row/column.

        Semantics for reserved ions match your core-based version:
        - reserved: BT fixes x/t/w; no one-hot on x/t; y[r,k,c_fix,i] <-> p[r,k,i],
            y[r,k,c ≠ c_fix,i] = 0.
        - free: x/t one-hot; y[r,k,c,i] <-> x[r,i,c] & p[r,k,i].
        """

        DEBUG_DIAG = True

        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)

        if BT is None:
            BT = [{} for _ in range(R)]

        # -------------------------------
        # Pre-checks (same semantics as before)
        # -------------------------------
        row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}
        ions_all = set(int(x) for x in A_in.flatten())

        # (a) No two ions pinned to the same (d,c) in a round
        for r, bt in enumerate(BT):
            seen = {}
            for i, (d, c) in bt.items():
                if i not in ions_all:
                    continue
                if (d, c) in seen:
                    raise ValueError(
                        f"UNSAT: BT[{r}] pins ions {seen[(d,c)]} and {i} "
                        f"to the same cell (d={d}, c={c})."
                    )

        # (b) Pair vs BT conflicts: same round, incompatible BT rows/blocks
        for r, pairs in enumerate(P_arr):
            for i1, i2 in pairs:
                if i1 not in ions_all or i2 not in ions_all:
                    continue
                if i1 in BT[r] and i2 in BT[r]:
                    d1, c1 = BT[r][i1]
                    d2, c2 = BT[r][i2]
                    if d1 != d2:
                        raise ValueError(
                            f"UNSAT: round {r} pair {(i1,i2)} BT rows differ: {d1} vs {d2}."
                        )
                    if (c1 // k) != (c2 // k):
                        raise ValueError(
                            f"UNSAT: round {r} pair {(i1,i2)} BT blocks differ: {c1}//{k} vs {c2}//{k}."
                        )

        # (c) Round-0: same source row & same target column among reserved ions
        if len(BT) >= 1:
            buckets = {}
            for i, (d0, c0) in BT[0].items():
                if i not in ions_all:
                    continue
                sr = row_of[i]
                buckets.setdefault((sr, c0), []).append(i)
            bad = {key: vs for key, vs in buckets.items() if len(vs) > 1}
            if bad:
                raise ValueError(
                    "UNSAT: round 0 has reserved ions from the same start row "
                    f"targeting the same column: {bad}"
                )

        # (d) Column oversubscription from BT
        for r, bt in enumerate(BT):
            col_counts = {}
            for i, (_, c) in bt.items():
                if i not in ions_all:
                    continue
                col_counts[c] = col_counts.get(c, 0) + 1
            bad_cols = {c: cnt for c, cnt in col_counts.items() if cnt > n}
            if bad_cols:
                raise ValueError(
                    f"UNSAT: BT[{r}] pins {bad_cols} ions to one column, exceeds n={n}."
                )

        ions = sorted(ions_all)
        num_blocks = math.ceil(m / k)

        # -------------------------------
        # Structural CNF builder
        # -------------------------------
        def build_structural_cnf(D_bound: int,
                                use_wcnf: bool = False,
                                add_boundary_soft: bool = False):
            """
            Build CNF (or WCNF) encoding:

            Vars:
                a[r,k,j,i] : layout at round r (r=0..R)
                x[r,i,c]   : dest column in round r
                t[r,i,d]   : dest row    in round r
                w[r,i,b]   : dest block  in round r
                p[r,k,i]   : ion i present in row k at layout a[r]
                y[r,k,c,i] : ion i in row k targeting col c at round r

            Hard:
                - a: exactly one ion per cell for r=0..R,
                - initial a[0] equals A_in,
                - x/t one-hot **only for non-reserved ions**,
                - w link only for non-reserved ions,
                - BT pins: x/t/w units + a[r+1,d_fix,c_fix,i],
                - pair equalities on t and w,
                - a[r+1] <-> (x[r], t[r]),
                - p & y as in your snippet (reserved vs free semantics),
                - bound: |c - j| <= D_bound and |d - k| <= D_bound.

            Soft:
                - (optional) boundary avoidance on last row/col for interacting ions.
            """

            vpool = IDPool()

            def var_a(r, k, j, i):  return vpool.id(('a', r, k, j, i))
            def var_x(r, i, c):     return vpool.id(('x', r, i, c))
            def var_t(r, i, d):     return vpool.id(('t', r, i, d))
            def var_w(r, i, b):     return vpool.id(('w', r, i, b))
            def var_p(r, k, i):     return vpool.id(('p', r, k, i))
            def var_y(r, k, c, i):  return vpool.id(('y', r, k, c, i))

            def is_reserved(r, i):  return i in BT[r]

            if use_wcnf:
                f = WCNF()
                def add_hard(cl): f.append(cl)
                def add_soft(cl, w): f.append(cl, weight=w)
            else:
                f = CNF()
                def add_hard(cl): f.append(cl)
                def add_soft(cl, w): raise RuntimeError("soft clauses not allowed in pure CNF")

            # (0) a-cardinality: exactly one ion per cell for r=0..R
            for r in range(R + 1):
                for krow in range(n):
                    for jcol in range(m):
                        lits = [var_a(r, krow, jcol, i) for i in ions]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_hard(cl)

            # (1) initial layout a[0]
            for krow in range(n):
                for jcol in range(m):
                    ion0 = int(A_in[krow, jcol])
                    add_hard([var_a(0, krow, jcol, ion0)])

            # (2) BT units and x/t/w-onehot for **free** ions
            for r in range(R):
                for i in ions:
                    if is_reserved(r, i):
                        # reserved: BT gives units, no equals on x/t/w
                        d_fix, c_fix = BT[r][i]
                        add_hard([var_x(r, i, c_fix)])
                        add_hard([var_t(r, i, d_fix)])
                        add_hard([var_w(r, i, c_fix // k)])
                        # fix layout after this round
                        add_hard([var_a(r + 1, d_fix, c_fix, i)])
                    else:
                        # free ion: x one-hot
                        lits_x = [var_x(r, i, c) for c in range(m)]
                        encx = CardEnc.equals(lits=lits_x, encoding=EncType.ladder, vpool=vpool)
                        for cl in encx.clauses:
                            add_hard(cl)
                        # free ion: t one-hot
                        lits_t = [var_t(r, i, d) for d in range(n)]
                        enct = CardEnc.equals(lits=lits_t, encoding=EncType.ladder, vpool=vpool)
                        for cl in enct.clauses:
                            add_hard(cl)

            # (3) w-link for free ions (reserved ions got w fixed above)
            for r in range(R):
                for i in ions:
                    if not is_reserved(r, i):
                        for b in range(num_blocks):
                            cols = list(range(b * k, min((b + 1) * k, m)))
                            wv = var_w(r, i, b)
                            add_hard([-wv] + [var_x(r, i, c) for c in cols])
                            for c in cols:
                                add_hard([-var_x(r, i, c), wv])

            # (4) pair constraints: same dest row & block
            for r in range(R):
                for (i1, i2) in P_arr[r]:
                    if i1 not in ions or i2 not in ions:
                        continue
                    for d in range(n):
                        add_hard([-var_t(r, i1, d), var_t(r, i2, d)])
                        add_hard([-var_t(r, i2, d), var_t(r, i1, d)])
                    for b in range(num_blocks):
                        add_hard([-var_w(r, i1, b), var_w(r, i2, b)])
                        add_hard([-var_w(r, i2, b), var_w(r, i1, b)])

            # (5) glue: a[r+1,d,c,i] <-> (x[r,i,c] & t[r,i,d]) for *all* ions
            for r in range(R):
                for i in ions:
                    for d in range(n):
                        for c in range(m):
                            a_next = var_a(r + 1, d, c, i)
                            xv = var_x(r, i, c)
                            tv = var_t(r, i, d)
                            add_hard([-a_next, xv])
                            add_hard([-a_next, tv])
                            add_hard([-xv, -tv, a_next])

            # (6) p row presence & y linkage (your semantics)
            for r in range(R):
                # (4a) p[r,k,i] <-> OR_j a[r,k,j,i]
                for krow in range(n):
                    for i in ions:
                        A_lits = [var_a(r, krow, j, i) for j in range(m)]
                        add_hard([-var_p(r, krow, i)] + A_lits)
                        for aj in A_lits:
                            add_hard([-aj, var_p(r, krow, i)])

                # (3) y at-most-1 and (4b) y-link
                for krow in range(n):
                    for c in range(m):
                        # y at-most-1 per (r,krow,c)
                        y_lits = [var_y(r, krow, c, i) for i in ions]
                        enc = CardEnc.atmost(lits=y_lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_hard(cl)

                        # y linkage, split free vs reserved
                        for i in ions:
                            yv = var_y(r, krow, c, i)
                            pv = var_p(r, krow, i)

                            if not is_reserved(r, i):
                                # free ion: y <-> (x & p)
                                xv = var_x(r, i, c)
                                add_hard([-yv, xv])      # y -> x
                                add_hard([-yv, pv])      # y -> p
                                add_hard([-xv, -pv, yv]) # (x & p) -> y
                            else:
                                # reserved ion
                                d_fix, c_fix = BT[r][i]
                                if c == c_fix:
                                    # y[r,k,c_fix,i] <-> p[r,k,i]
                                    add_hard([-yv, pv])   # y -> p
                                    add_hard([-pv, yv])   # p -> y
                                else:
                                    # y[r,k,c,i] = 0 for c != c_fix
                                    add_hard([-yv])

            # (7) movement bound: |c - j| <= D_bound, |d - k| <= D_bound
            if D_bound is not None:
                for r in range(R):
                    for krow in range(n):
                        for jcol in range(m):
                            for i in ions:
                                a_src = var_a(r, krow, jcol, i)
                                # horizontal distance
                                for c in range(m):
                                    if abs(c - jcol) > D_bound:
                                        add_hard([-a_src, -var_x(r, i, c)])
                                # vertical distance
                                for d in range(n):
                                    if abs(d - krow) > D_bound:
                                        add_hard([-a_src, -var_t(r, i, d)])

            # (8) optional soft boundary-avoidance
            if use_wcnf and add_boundary_soft:
                for r in range(R):
                    for (i1, i2) in P_arr[r]:
                        if i1 in ions:
                            add_soft([-var_x(r, i1, m - 1)], wB_col)
                            add_soft([-var_t(r, i1, n - 1)], wB_row)
                        if i2 in ions:
                            add_soft([-var_x(r, i2, m - 1)], wB_col)
                            add_soft([-var_t(r, i2, n - 1)], wB_row)

            return f, vpool, ions, var_a, var_x, var_t

        # -------------------------------
        # Level 1: SAT + binary search on D
        # -------------------------------
        D_lo = 0
        D_hi = max(n - 1, m - 1)
        best_D = None

        if DEBUG_DIAG:
            print(f"[WISE] starting binary search for D in [0, {D_hi}]")

        while D_lo <= D_hi:
            D_mid = (D_lo + D_hi) // 2
            cnf_mid, vpool_mid, ions_mid, var_a_mid, var_x_mid, var_t_mid = \
                build_structural_cnf(D_mid, use_wcnf=False, add_boundary_soft=False)

            t_sat_start = time.time()
            with Minisat22(bootstrap_with=cnf_mid.clauses) as sat:
                sat_ok = sat.solve()
                model_mid = sat.get_model() if sat_ok else None
            t_sat_end = time.time()

            if DEBUG_DIAG:
                print(f"[WISE]  test D={D_mid}: SAT={sat_ok}, "
                    f"vars={vpool_mid.top}, clauses={len(cnf_mid.clauses)}, "
                    f"time={t_sat_end - t_sat_start:.3f}s")

            if sat_ok:
                best_D = (D_mid, model_mid, vpool_mid, ions_mid, var_a_mid)
                D_hi = D_mid - 1
            else:
                D_lo = D_mid + 1

        if best_D is None:
            raise RuntimeError("No feasible layout for any D in [0, max(n-1,m-1)].")

        D_star, _, _, _, _ = best_D
        if DEBUG_DIAG:
            print(f"[WISE] minimal D* found: {D_star}")

        # -------------------------------
        # Level 2: MaxSAT at D* (boundary avoidance)
        # -------------------------------
        wcnf, vpool, ions_mid, var_a, var_x, var_t = \
            build_structural_cnf(D_star, use_wcnf=True, add_boundary_soft=True)

        if DEBUG_DIAG:
            print(f"[WISE] WCNF at D*: vars={wcnf.nv}, hard={len(wcnf.hard)}, soft={len(wcnf.soft)}")

        t_rc2_start = time.time()
        rc2 = RC2(wcnf)
        model = rc2.compute()
        t_rc2_end = time.time()

        if model is None:
            raise RuntimeError("MaxSAT at D* unexpectedly UNSAT.")

        if DEBUG_DIAG:
            print(f"[WISE] RC2 at D*: time={t_rc2_end - t_rc2_start:.3f}s, opt_cost={rc2.cost}")

        model_set = set(l for l in model if l > 0)
        def lit_true(v: int) -> bool: return v in model_set

        # -------------------------------
        # Decode layouts a[1]..a[R]
        # -------------------------------
        layouts: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            rr = r + 1  # layout after round r
            for d in range(n):
                for c in range(m):
                    found = None
                    for i in ions_mid:
                        if lit_true(var_a(rr, d, c, i)):
                            found = i
                            break
                    if found is None:
                        raise RuntimeError(f"could not reconstruct cell (round={rr}, d={d}, c={c})")
                    nxt[d, c] = found
            layouts.append(nxt)
            cur = nxt

        return layouts


    @staticmethod
    def _optimal_QMR_for_WISE5(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,
        wV: List[int] = None,
        wB_col: int = 1,
        wB_row: int = 1
    ) -> List[np.ndarray]:
        """
        Max-SAT layout optimizer for WISE on a slice.

        Semantics:
        - BT pins for *interior* cells of the current slice are hard units.
        - BT pins for *frontier* cells (last row or last col of this A_in) are guarded
            by selector assumptions; if they conflict with 6b/pairs/etc they are dropped.
        - 6b (q) enforces exact destination-cell uniqueness **only for the last round**.
        - u-helpers are removed.
        - y is at-most-1 per (r,k,c); exactness per destination cell comes from q on r=R-1.
        - Horizontal / vertical band linkage (11),(12) is applied only to the last
            MAX_BAND_ROUNDS rounds in this call (to keep clause blow-up under control).
        """

        DEBUG_DIAG = True          # flip to False to silence diagnostics
        MAX_BAND_ROUNDS = 1        # charge bands only for last this-many rounds in this call

        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)

        # weights
        if wH is None:
            wH = [1 + d for d in range(m)]
        if wV is None:
            wV = [1 + d for d in range(n)]
        if BT is None:
            BT = [{} for _ in range(R)]

        # Ions present in THIS slice
        ions_in_slice = set(int(x) for x in A_in.flatten())

        # frontier test for this slice
        def is_frontier_cell(d: int, c: int) -> bool:
            # frontier = last row OR last col of current A_in
            return (d == n - 1) or (c == m - 1)

        t_build_start = time.time()

        # -------------------------------
        # Cheap structural prechecks (restricted to ions in this slice)
        # -------------------------------
        row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}

        # (a) No two *present* ions pinned to the same (d,c) in a round
        for r, bt in enumerate(BT):
            seen = {}
            for i, (d, c) in bt.items():
                if i not in ions_in_slice:
                    continue
                if (d, c) in seen:
                    raise ValueError(
                        f"UNSAT: BT[{r}] pins ions {seen[(d,c)]} and {i} "
                        f"to the same cell (d={d}, c={c})."
                    )

        # (b) Pair vs BT conflicts: same round, incompatible BT rows/blocks
        for r, pairs in enumerate(P_arr):
            for i1, i2 in pairs:
                if (i1 not in ions_in_slice) or (i2 not in ions_in_slice):
                    continue
                if i1 in BT[r] and i2 in BT[r]:
                    d1, c1 = BT[r][i1]
                    d2, c2 = BT[r][i2]
                    if d1 != d2:
                        raise ValueError(
                            f"UNSAT: round {r} pair {(i1,i2)} BT rows differ: {d1} vs {d2}."
                        )
                    if (c1 // k) != (c2 // k):
                        raise ValueError(
                            f"UNSAT: round {r} pair {(i1,i2)} BT blocks differ: {c1}//{k} vs {c2}//{k}."
                        )

        # (c) Round-0: same *source row* & same target column among reserved ions IN SLICE
        if len(BT) >= 1:
            buckets = {}
            for i, (d0, c0) in BT[0].items():
                if i not in ions_in_slice:
                    continue
                sr = row_of[i]
                buckets.setdefault((sr, c0), []).append(i)
            bad = {key: vs for key, vs in buckets.items() if len(vs) > 1}
            if bad:
                raise ValueError(
                    "UNSAT: round 0 has reserved ions from the same start row "
                    f"targeting the same column: {bad}"
                )

        # (d) Column oversubscription from BT (only ions in this slice matter)
        for r, bt in enumerate(BT):
            col_counts = {}
            for i, (_, c) in bt.items():
                if i not in ions_in_slice:
                    continue
                col_counts[c] = col_counts.get(c, 0) + 1
            bad_cols = {c: cnt for c, cnt in col_counts.items() if cnt > n}
            if bad_cols:
                raise ValueError(
                    f"UNSAT: BT[{r}] pins {bad_cols} ions to one column, exceeds n={n}."
                )

        # (e) Row capacity precheck for 6b (approximate, restricted to ions in this slice)
        from collections import defaultdict, deque
        all_ions = set(ions_in_slice)

        for r in range(R):
            # Build pair graph
            G = defaultdict(set)
            for (i1, i2) in P_arr[r]:
                if i1 not in all_ions or i2 not in all_ions:
                    continue
                G[i1].add(i2)
                G[i2].add(i1)

            # Connected components
            seen = set()
            comps = []
            for i in all_ions:
                if i in seen:
                    continue
                q = deque([i])
                comp = []
                seen.add(i)
                while q:
                    u = q.popleft()
                    comp.append(u)
                    for v in G[u]:
                        if v not in seen:
                            seen.add(v)
                            q.append(v)
                comps.append(comp)

            # Row forces from BT within a component
            forced_row = {}
            for comp in comps:
                pin_rows = {BT[r][i][0] for i in comp if i in BT[r]}
                if len(pin_rows) > 1:
                    raise RuntimeError(
                        f"UNSAT precheck: round {r} a pair-component has conflicting BT rows {pin_rows}"
                    )
                if len(pin_rows) == 1:
                    d_fix = pin_rows.pop()
                    for i in comp:
                        forced_row[i] = d_fix

            allow = [0] * n
            for i in all_ions:
                if i in BT[r]:
                    allow[BT[r][i][0]] += 1
                elif i in forced_row:
                    allow[forced_row[i]] += 1

            free_unforced = [i for i in all_ions if i not in BT[r] and i not in forced_row]

            bad = [(d, allow[d]) for d in range(n) if allow[d] > m]
            if bad:
                raise RuntimeError(
                    f"UNSAT precheck: round {r} row over-subscribed by BT/pairs: {bad} (> m={m})."
                )

            bad_need = [(d, allow[d]) for d in range(n) if allow[d] + len(free_unforced) < m]
            if bad_need:
                raise RuntimeError(
                    f"UNSAT precheck: round {r} row lacks candidates for (6b) exactness: "
                    f"{bad_need}; free_unforced={len(free_unforced)}, m={m}"
                )

        # -------------------------------
        # SAT encoding
        # -------------------------------
        vpool = IDPool()

        # variable accessors
        def var_x(r, i, c): return vpool.id(('x', r, i, c))
        def var_t(r, i, d): return vpool.id(('t', r, i, d))
        def var_w(r, i, b): return vpool.id(('w', r, i, b))
        def var_a(r, krow, jcol, i): return vpool.id(('a', r, krow, jcol, i))
        def var_p(r, krow, i): return vpool.id(('p', r, krow, i))
        def var_y(r, krow, c, i): return vpool.id(('y', r, krow, c, i))
        def var_h(r, delta): return vpool.id(('h', r, delta))
        def var_v(r, delta): return vpool.id(('v', r, delta))
        def var_q(r, d, c, i): return vpool.id(('q', r, d, c, i))

        ions: List[int] = sorted(ions_in_slice)

        def is_interior_pinned(r, ion) -> bool:
            if ion not in BT[r]:
                return False
            d, c = BT[r][ion]
            return not is_frontier_cell(d, c)

        def is_frontier_pinned(r, ion) -> bool:
            if ion not in BT[r]:
                return False
            d, c = BT[r][ion]
            return is_frontier_cell(d, c)

        # Base hard CNF (no frontier BT yet)
        hard_base = CNF()

        def add_hard(clause):
            hard_base.append(clause)

        # For diagnostics: track clause counts after each major block
        marks = []

        def mark(label: str):
            marks.append((label, len(hard_base.clauses)))

        # Frontier BT guards: (selector_var, literal)
        bt_guards: List[Tuple[int, int]] = []
        assumps: List[int] = []

        def add_frontier_bt_unit(sel_key, litpos: int):
            """
            sel_key: tuple describing this BT (for debugging)
            litpos: positive variable that we want as a unit IF this BT is kept.
            """
            s = vpool.id(('selBT',) + sel_key)
            bt_guards.append((s, litpos))
            assumps.append(s)

        # (0) exactly one ion per cell of A[r]: for all r,k,j: sum_i a[r,k,j,i] = 1
        # Here, r = 0..R-2 for 'a', because the last layout uses q (r=R-1).
        for r in range(max(1, R)):  # at least r=0 exists
            if r >= R:
                break
            for krow in range(n):
                for jcol in range(m):
                    lits = [var_a(r, krow, jcol, i) for i in ions]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl)
        mark("(0) a-cells")

        # (1) initial layout at r=0
        for krow in range(n):
            for jcol in range(m):
                ion = int(A_in[krow, jcol])
                add_hard([var_a(0, krow, jcol, ion)])
        mark("(1) init")

        # (2) BT-reserved ions: interior pins become hard units; frontier pins become guarded
        for r in range(R):
            for ion in ions:
                if ion in BT[r]:
                    d_fix, c_fix = BT[r][ion]
                    interior = not is_frontier_cell(d_fix, c_fix)

                    vx = var_x(r, ion, c_fix)
                    vt = var_t(r, ion, d_fix)
                    vw = var_w(r, ion, c_fix // k)

                    if interior:
                        add_hard([vx])
                        add_hard([vt])
                        add_hard([vw])
                    else:
                        add_frontier_bt_unit(('x', r, ion, c_fix), vx)
                        add_frontier_bt_unit(('t', r, ion, d_fix), vt)
                        add_frontier_bt_unit(('w', r, ion, c_fix // k), vw)

                    # glue into next round if exists (this is about A[r+1,...], not frontier)
                    if r < R - 1:
                        add_hard([var_a(r + 1, d_fix, c_fix, ion)])
        mark("(2) BT")

        # (3) Row–column mapping: y at-most-1 per (r,k,c), x-onehot for ions not interior-pinned
        for r in range(R):
            # y at-most-1 per (k,c)
            for krow in range(n):
                for c in range(m):
                    lits = [var_y(r, krow, c, i) for i in ions]
                    enc = CardEnc.atmost(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl)

            # x one-hot for ions that are not interior-pinned
            for ion in ions:
                if not is_interior_pinned(r, ion):
                    lits = [var_x(r, ion, c) for c in range(m)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl)
        mark("(3) y-atmost-1 + x-onehot")

        # (4) Row presence p and y linkage
        for r in range(R):
            # p[r,k,i] <-> OR_j a[r,k,j,i]
            for krow in range(n):
                for ion in ions:
                    A_lits = [var_a(r, krow, jcol, ion) for jcol in range(m)]
                    add_hard([-var_p(r, krow, ion)] + A_lits)
                    for aj in A_lits:
                        add_hard([-aj, var_p(r, krow, ion)])

            # y linkage:
            #   interior-pinned: y[r,k,c_fix,i] <-> p[r,k,i], y=0 elsewhere
            #   others (including frontier-pinned & free): y <-> (x & p)
            for krow in range(n):
                for c in range(m):
                    for ion in ions:
                        yv = var_y(r, krow, c, ion)
                        pv = var_p(r, krow, ion)
                        if is_interior_pinned(r, ion):
                            d_fix, c_fix = BT[r][ion]
                            if c == c_fix:
                                add_hard([-yv, pv])   # y -> p
                                add_hard([-pv, yv])   # p -> y
                            else:
                                add_hard([-yv])       # y = 0 at other columns
                        else:
                            xv = var_x(r, ion, c)
                            add_hard([-yv, xv])
                            add_hard([-yv, pv])
                            add_hard([-xv, -pv, yv])
        mark("(4) p-row + y-link")

        # (5) destination-row unique (t one-hot) for ions that are not interior-pinned
        for r in range(R):
            for ion in ions:
                if not is_interior_pinned(r, ion):
                    lits = [var_t(r, ion, d) for d in range(n)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl)
        mark("(5) t-onehot")

        # (6b) destination-cell exactness via q **only for the last round r_last = R-1**:
        #     interior-pinned: q is hard unit at pinned cell, false elsewhere
        #     others (including frontier-pinned & free): q <-> (x & t)
        if R > 0:
            r_last = R - 1
            for d in range(n):
                for c in range(m):
                    q_lits = []
                    for ion in ions:
                        qv = var_q(r_last, d, c, ion)
                        q_lits.append(qv)
                        if is_interior_pinned(r_last, ion):
                            d_fix, c_fix = BT[r_last][ion]
                            if d == d_fix and c == c_fix:
                                add_hard([qv])
                            else:
                                add_hard([-qv])
                        else:
                            xv = var_x(r_last, ion, c)
                            tv = var_t(r_last, ion, d)
                            add_hard([-qv, xv])
                            add_hard([-qv, tv])
                            add_hard([-xv, -tv, qv])
                    enc = CardEnc.equals(lits=q_lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl)
        mark("(6b) q-exact-last-round")

        # (7) block membership w for ions that are not interior-pinned
        num_blocks = math.ceil(m / k)
        for r in range(R):
            for ion in ions:
                if not is_interior_pinned(r, ion):
                    for b in range(num_blocks):
                        cols = list(range(b * k, min((b + 1) * k, m)))
                        wv = var_w(r, ion, b)
                        add_hard([-wv] + [var_x(r, ion, c) for c in cols])
                        for c in cols:
                            add_hard([-var_x(r, ion, c), wv])
        mark("(7) w-blocks")

        # (8) pair constraints: same dest row (t) and same block (w)
        for r in range(R):
            for (i1, i2) in P_arr[r]:
                if i1 not in ions_in_slice or i2 not in ions_in_slice:
                    continue
                # same dest row
                for d in range(n):
                    add_hard([-var_t(r, i1, d), var_t(r, i2, d)])
                    add_hard([-var_t(r, i2, d), var_t(r, i1, d)])
                # same block
                for b in range(num_blocks):
                    add_hard([-var_w(r, i1, b), var_w(r, i2, b)])
                    add_hard([-var_w(r, i2, b), var_w(r, i1, b)])
        mark("(8) pair-equalities")

        # (9) rounds glued together for ions that are not interior-pinned:
        #     a[r+1, d, c, i] <-> (x[r, i, c] & t[r, i, d])
        for r in range(R - 1):
            for ion in ions:
                if not is_interior_pinned(r, ion):
                    for c in range(m):
                        for d in range(n):
                            av = var_a(r + 1, d, c, ion)
                            xv = var_x(r, ion, c)
                            tv = var_t(r, ion, d)
                            add_hard([-av, xv])
                            add_hard([-av, tv])
                            add_hard([-xv, -tv, av])
                # interior-pinned ions already have a[r+1,...] fixed in (2)
        mark("(9) glue a<->x,t")

        # (10) band monotonicity
        for r in range(R):
            for delta in range(m - 1):
                add_hard([-var_h(r, delta), var_h(r, delta + 1)])
            for delta in range(n - 1):
                add_hard([-var_v(r, delta), var_v(r, delta + 1)])
        mark("(10) band-monotone")

        # (11) horizontal linkage (only ions not interior-pinned),
        #      BUT only for last MAX_BAND_ROUNDS rounds to limit blow-up
        for r in range(R):
            # if r < max(0, R - MAX_BAND_ROUNDS):
            #     continue
            for krow in range(n):
                for jcol in range(m):
                    for c in range(m):
                        dist = abs(c - jcol)
                        hv = var_h(r, dist)
                        for ion in ions:
                            if not is_interior_pinned(r, ion):
                                add_hard([
                                    -var_y(r, krow, c, ion),
                                    -var_a(r, krow, jcol, ion),
                                    hv
                                ])
        mark("(11) h-link-last-rounds")

        # (12) vertical linkage (only ions not interior-pinned),
        #      also limited to last MAX_BAND_ROUNDS rounds
        for r in range(R):
            # if r < max(0, R - MAX_BAND_ROUNDS):
            #     continue
            for krow in range(n):
                for d in range(n):
                    dist = abs(d - krow)
                    vv = var_v(r, dist)
                    for ion in ions:
                        if not is_interior_pinned(r, ion):
                            add_hard([
                                -var_t(r, ion, d),
                                -var_p(r, krow, ion),
                                vv
                            ])
        mark("(12) v-link-last-rounds")

        t_build_end = time.time()

        # Frontier BT counts (for diagnostics)
        frontier_bt_count = 0
        for r in range(R):
            for ion in ions:
                if is_frontier_pinned(r, ion):
                    frontier_bt_count += 1

        if DEBUG_DIAG:
            print(f"[WISE] slice: n={n}, m={m}, R={R}, ions_in_slice={len(ions)}")
            print(f"[WISE] hard_base built in {t_build_end - t_build_start:.3f}s; "
                f"vars={vpool.top}, clauses={len(hard_base.clauses)}; "
                f"frontier BT pins={frontier_bt_count}")
            # Clause breakdown
            print("[WISE] clause breakdown (incremental):")
            prev = 0
            for label, cnt in marks:
                print(f"  {label}: +{cnt - prev} (total {cnt})")
                prev = cnt

        # ------------------------------------------------
        # HARD PRE-SOLVE with frontier BT as assumptions
        # ------------------------------------------------
        hard_for_presolve = CNF()
        for cl in hard_base.clauses:
            hard_for_presolve.append(list(cl))
        for s, lit in bt_guards:
            hard_for_presolve.append([-s, lit])

        if bt_guards:
            t_pre_start = time.time()
            with Minisat22(bootstrap_with=hard_for_presolve.clauses) as sat:
                ok = sat.solve(assumptions=assumps)
                if ok:
                    core = None
                else:
                    core = set(sat.get_core())
            t_pre_end = time.time()
            if DEBUG_DIAG:
                print(f"[WISE] presolve: SAT={ok}, "
                    f"selectors={len(bt_guards)}, "
                    f"core_size={0 if core is None else len(core)}, "
                    f"time={t_pre_end - t_pre_start:.3f}s")
        else:
            core = None

        bad_selectors = set(core) if core is not None else set()
        good_selectors = [s for s in assumps if s not in bad_selectors]

        # ------------------------------------------------
        # Build WCNF for RC2: hard + surviving frontier BT + soft
        # ------------------------------------------------
        wcnf = WCNF()

        # base hard clauses
        for cl in hard_base.clauses:
            wcnf.append(list(cl))

        # surviving frontier BT units promoted to hard
        good_set = set(good_selectors)
        for s, lit in bt_guards:
            if s in good_set:
                wcnf.append([lit])  # hard unit

        # Soft constraints: band minimization + boundary avoidance
        for r in range(R):
            # horizontal bands
            for delta in range(m):
                wcnf.append([-var_h(r, delta)], weight=wH[delta])
            # vertical bands
            for delta in range(n):
                wcnf.append([-var_v(r, delta)], weight=wV[delta])
            # boundary avoidance for pairs (only ions not interior-pinned)
            for (i1, i2) in P_arr[r]:
                if i1 in ions_in_slice and not is_interior_pinned(r, i1):
                    wcnf.append([-var_x(r, i1, m - 1)], weight=wB_col)
                    wcnf.append([-var_t(r, i1, n - 1)], weight=wB_row)
                if i2 in ions_in_slice and not is_interior_pinned(r, i2):
                    wcnf.append([-var_x(r, i2, m - 1)], weight=wB_col)
                    wcnf.append([-var_t(r, i2, n - 1)], weight=wB_row)

        if DEBUG_DIAG:
            n_hard = len(wcnf.hard)
            n_soft = len(wcnf.soft)
            print(f"[WISE] WCNF built: vars={wcnf.nv}, hard={n_hard}, soft={n_soft}")

        # Solve MaxSAT
        t_rc2_start = time.time()
        solver = RC2(wcnf)
        model = solver.compute()
        t_rc2_end = time.time()

        if DEBUG_DIAG:
            try:
                opt_cost = solver.cost
            except AttributeError:
                opt_cost = None
            print(f"[WISE] RC2 solve: time={t_rc2_end - t_rc2_start:.3f}s, "
                f"opt_cost={opt_cost}")

        if model is None:
            raise RuntimeError("UNSAT (hard) after dropping conflicting frontier BT pins")

        model_set = set(l for l in model if l > 0)

        def lit_true(v: int) -> bool:
            return v in model_set

        # ------------------------------------------------
        # Decode: reconstruct A_1..A_R
        # ------------------------------------------------
        layouts: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            if r < R - 1:
                # use a[r+1,d,c,i]
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions:
                            if lit_true(var_a(r + 1, d, c, ion)):
                                found = ion
                                break
                        if found is None:
                            raise RuntimeError(
                                f"could not reconstruct cell (round={r+1}, d={d}, c={c})"
                            )
                        nxt[d, c] = found
            else:
                # last round: use q[r,d,c,i] from 6b
                r_last = R - 1
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions:
                            if lit_true(var_q(r_last, d, c, ion)):
                                found = ion
                                break
                        if found is None:
                            raise RuntimeError(
                                f"could not reconstruct final cell (round={r_last}, d={d}, c={c})"
                            )
                        nxt[d, c] = found

            layouts.append(nxt)
            cur = nxt

        return layouts





    @staticmethod
    def _optimal_QMR_for_WISE4(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,
        wV: List[int] = None,
        wB_col: int = 1,
        wB_row: int = 1
    ) -> List[np.ndarray]:
        # -------------------------
        # Helper for UNSAT core
        # -------------------------
        class CoreBuilder:
            def __init__(self, vpool: IDPool):
                self.vpool = vpool
                self.cnf = CNF()
                self.sel_by_name = {}    # name -> selector var (positive)
                self.assumps = []        # list of selectors (positive)

            def selector(self, name: str) -> int:
                if name not in self.sel_by_name:
                    s = self.vpool.id(('sel', name))
                    self.sel_by_name[name] = s
                    self.assumps.append(s)
                return self.sel_by_name[name]

            def add(self, name: str, clause):
                s = self.selector(name)
                self.cnf.append([-s] + clause)

            def extend(self, name: str, clauses):
                s = self.selector(name)
                for cl in clauses:
                    self.cnf.append([-s] + cl)

            def build_and_solve(self):
                with Minisat22(bootstrap_with=self.cnf.clauses) as sat:
                    ok = sat.solve(assumptions=self.assumps)
                    if ok:
                        return None
                    core = sat.get_core()
                inv = {v: name for name, v in self.sel_by_name.items()}
                names = sorted(inv[v] for v in core)
                return names

        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)

        # weights
        if wH is None:
            wH = [1 + d for d in range(m)]
        if wV is None:
            wV = [1 + d for d in range(n)]
        if BT is None:
            BT = [{} for _ in range(R)]

        # quick pre-checks you already had (kept)
        row_of = {int(A_in[r,c]): r for r in range(n) for c in range(m)}
        for r, bt in enumerate(BT):
            seen = {}
            for i, (d, c) in bt.items():
                if (d, c) in seen:
                    raise ValueError(f"UNSAT: BT[{r}] pins ions {seen[(d,c)]} and {i} to the same cell (d={d}, c={c}).")
        for r, pairs in enumerate(P_arr):
            for i1, i2 in pairs:
                if i1 in BT[r] and i2 in BT[r]:
                    d1, c1 = BT[r][i1]; d2, c2 = BT[r][i2]
                    if d1 != d2:
                        raise ValueError(f"UNSAT: round {r} pair {(i1,i2)} BT rows differ: {d1} vs {d2}.")
                    if (c1 // k) != (c2 // k):
                        raise ValueError(f"UNSAT: round {r} pair {(i1,i2)} BT blocks differ: {c1}//{k} vs {c2}//{k}.")
        if len(BT) >= 1:
            buckets = {}
            for i, (d0, c0) in BT[0].items():
                sr = row_of[i]
                buckets.setdefault((sr, c0), []).append(i)
            bad = {key: vs for key, vs in buckets.items() if len(vs) > 1}
            if bad:
                raise ValueError(f"UNSAT: round 0 has reserved ions from the same start row targeting the same column: {bad}")
        # column oversubscription from BT
        for r, bt in enumerate(BT):
            col_counts = {}
            for i, (_, c) in bt.items():
                col_counts[c] = col_counts.get(c, 0) + 1
            bad_cols = {c: cnt for c, cnt in col_counts.items() if cnt > n}
            if bad_cols:
                raise ValueError(f"UNSAT: BT[{r}] pins {bad_cols} ions to one column, exceeds n={n}.")

        vpool = IDPool()

        # -------------------------
        # variable accessors
        # -------------------------
        def var_x(r, i, c): return vpool.id(('x', r, i, c))
        def var_t(r, i, d): return vpool.id(('t', r, i, d))
        def var_w(r, i, b): return vpool.id(('w', r, i, b))
        def var_a(r, krow, jcol, i): return vpool.id(('a', r, krow, jcol, i))
        def var_p(r, krow, i): return vpool.id(('p', r, krow, i))
        def var_y(r, krow, c, i): return vpool.id(('y', r, krow, c, i))
        def var_h(r, delta): return vpool.id(('h', r, delta))
        def var_v(r, delta): return vpool.id(('v', r, delta))
        def var_u(r, c, d, krow): return vpool.id(('u', r, c, d, krow))
        def var_q(r, d, c, i): return vpool.id(('q', r, d, c, i))

        ions: List[int] = sorted(int(x) for x in A_in.flatten())

        wcnf = WCNF()

        def is_reserved(r, ion): return ion in BT[r]

        # (0) exactly one ion per cell
        for r in range(R):
            for krow in range(n):
                for jcol in range(m):
                    lits = [var_a(r, krow, jcol, i) for i in ions]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses: wcnf.append(cl)

        # (1) initial layout
        for krow in range(n):
            for jcol in range(m):
                ion = int(A_in[krow, jcol])
                wcnf.append([var_a(0, krow, jcol, ion)])

        # (2) unit clauses for reserved ions
        for r in range(R):
            for ion in ions:
                if is_reserved(r, ion):
                    d, c = BT[r][ion]
                    wcnf.append([var_x(r, ion, c)])
                    wcnf.append([var_t(r, ion, d)])
                    b = c // k
                    wcnf.append([var_w(r, ion, b)])
                    if r < R - 1:
                        wcnf.append([var_a(r + 1, d, c, ion)])

        # # NEW reserved-propagation
        for r in range(R - 1):
            for ion in ions:
                if is_reserved(r, ion) and is_reserved(r + 1, ion):
                    d1, c1 = BT[r][ion]
                    d2, c2 = BT[r + 1][ion]
                    wcnf.append([var_p(r + 1, d1, ion)])
                    wcnf.append([var_y(r + 1, d1, c2, ion)])
                    wcnf.append([var_u(r + 1, c2, d2, d1)])

        # (3) row–column mapping
        for r in range(R):
            for krow in range(n):
                for c in range(m):
                    lits = [var_y(r, krow, c, i) for i in ions]
                    enc = CardEnc.atmost(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses: wcnf.append(cl)
            for ion in ions:
                if not is_reserved(r, ion):
                    lits = [var_x(r, ion, c) for c in range(m)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses: wcnf.append(cl)

        # (4) row presence + y linkage
        for r in range(R):
            for krow in range(n):
                for ion in ions:
                    A_lits = [var_a(r, krow, jcol, ion) for jcol in range(m)]
                    wcnf.append([-var_p(r, krow, ion)] + A_lits)
                    for aj in A_lits: wcnf.append([-aj, var_p(r, krow, ion)])

            for krow in range(n):
                for c in range(m):
                    for ion in ions:
                        yv = var_y(r, krow, c, ion)
                        pv = var_p(r, krow, ion)

                        if not is_reserved(r, ion):
                            xv = var_x(r, ion, c)
                            # y <-> (x & p)
                            wcnf.append([-yv, xv])
                            wcnf.append([-yv, pv])
                            wcnf.append([-xv, -pv, yv])
                        else:
                            d_fix, c_fix = BT[r][ion]
                            if c == c_fix:
                                # y <-> p  (and x is already unit-true from (2))
                                wcnf.append([-yv, pv])
                                wcnf.append([-pv, yv])
                            else:
                                # y must be false at all other columns
                                wcnf.append([-yv])

        # (5) destination-row unique (free)
        for r in range(R):
            for ion in ions:
                if not is_reserved(r, ion):
                    lits = [var_t(r, ion, d) for d in range(n)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses: wcnf.append(cl)

        # (6) column–dest-row uniqueness (cheap)
        for r in range(R):
            for c in range(m):
                for d in range(n):
                    u_lits = [var_u(r, c, d, krow) for krow in range(n)]
                    enc = CardEnc.equals(lits=u_lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses: wcnf.append(cl)
                    for krow in range(n):
                        uvar = var_u(r, c, d, krow)
                        ors_first, ors_second = [], []
                        for ion in ions:
                            if not is_reserved(r, ion):
                                ors_first.append(var_y(r, krow, c, ion))
                                ors_second.append(var_t(r, ion, d))
                        for ion in ions:
                            if is_reserved(r, ion):
                                d_fix, c_fix = BT[r][ion]
                                if d_fix == d and c_fix == c:
                                    ors_first.append(var_x(r, ion, c))
                                    ors_second.append(var_p(r, krow, ion))
                        if not ors_first:
                            wcnf.append([-uvar])
                        else:
                            wcnf.append([-uvar] + ors_first)
                            wcnf.append([-uvar] + ors_second)

       # (6b) destination-cell exactness
        for r in range(R):
            for d in range(n):
                for c in range(m):
                    q_lits = []
                    for ion in ions:
                        qv = var_q(r, d, c, ion)
                        q_lits.append(qv)

                        if not is_reserved(r, ion):
                            xv = var_x(r, ion, c)
                            tv = var_t(r, ion, d)
                            # q <-> (x & t)
                            wcnf.append([-qv, xv])
                            wcnf.append([-qv, tv])
                            wcnf.append([-xv, -tv, qv])
                        else:
                            d_fix, c_fix = BT[r][ion]
                            if d == d_fix and c == c_fix:
                                # this ion occupies exactly its pinned cell in round r
                                wcnf.append([qv])              # q[r,d*,c*,i] = True
                                # (x[t], t[t]) are already unit-True from (2)
                            else:
                                wcnf.append([-qv])             # q[r,d,c,i] = False

                    # exactly one ion per destination cell (r,d,c)
                    enc = CardEnc.equals(lits=q_lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)

        # (7) block membership
        num_blocks = math.ceil(m / k)
        for r in range(R):
            for ion in ions:
                if not is_reserved(r, ion):
                    for b in range(num_blocks):
                        cols = list(range(b * k, min((b + 1) * k, m)))
                        lits = [var_x(r, ion, c) for c in cols]
                        wv = var_w(r, ion, b)
                        wcnf.append([-wv] + lits)
                        for lc in lits: wcnf.append([-lc, wv])

        # (8) pair constraints
        for r in range(R):
            for (i1, i2) in P_arr[r]:
                for d in range(n):
                    wcnf.append([-var_t(r, i1, d), var_t(r, i2, d)])
                    wcnf.append([-var_t(r, i2, d), var_t(r, i1, d)])
                for b in range(num_blocks):
                    wcnf.append([-var_w(r, i1, b), var_w(r, i2, b)])
                    wcnf.append([-var_w(r, i2, b), var_w(r, i1, b)])

        # (9) rounds glued together (non-reserved)
        for r in range(R - 1):
            for ion in ions:
                if not is_reserved(r, ion):
                    for c in range(m):
                        for d in range(n):
                            av = var_a(r + 1, d, c, ion)
                            xv = var_x(r, ion, c)
                            tv = var_t(r, ion, d)
                            wcnf.append([-av, xv])
                            wcnf.append([-av, tv])
                            wcnf.append([-xv, -tv, av])

        # (10) band monotonicity (soft-only relevance; still add as hard)
        for r in range(R):
            for delta in range(m - 1):
                wcnf.append([-var_h(r, delta), var_h(r, delta + 1)])
            for delta in range(n - 1):
                wcnf.append([-var_v(r, delta), var_v(r, delta + 1)])

        # (11) horizontal linkage (only non-reserved)
        for r in range(R):
            for krow in range(n):
                for jcol in range(m):
                    for c in range(m):
                        dist = abs(c - jcol)
                        for ion in ions:
                            if not is_reserved(r, ion):
                                wcnf.append([-var_y(r, krow, c, ion), -var_a(r, krow, jcol, ion), var_h(r, dist)])

        # (12) vertical linkage (only non-reserved)
        for r in range(R):
            for krow in range(n):
                for d in range(n):
                    dist = abs(d - krow)
                    for ion in ions:
                        if not is_reserved(r, ion):
                            wcnf.append([-var_t(r, ion, d), -var_p(r, krow, ion), var_v(r, dist)])

        # Soft constraints
        for r in range(R):
            for delta in range(m):
                wcnf.append([-var_h(r, delta)], weight=wH[delta])
            for delta in range(n):
                wcnf.append([-var_v(r, delta)], weight=wV[delta])
            for (i1, i2) in P_arr[r]:
                if not is_reserved(r, i1):
                    wcnf.append([-var_x(r, i1, m - 1)], weight=wB_col)
                    wcnf.append([-var_t(r, i1, n - 1)], weight=wB_row)
                if not is_reserved(r, i2):
                    wcnf.append([-var_x(r, i2, m - 1)], weight=wB_col)
                    wcnf.append([-var_t(r, i2, n - 1)], weight=wB_row)

        for r in range(R):
            # Build a graph of equal-row constraints from pairs
            from collections import defaultdict, deque
            G = defaultdict(set)
            for (i1, i2) in P_arr[r]:
                G[i1].add(i2); G[i2].add(i1)

            # Find connected components of the pair-graph
            seen = set()
            comps = []
            for i in ions:
                if i in seen: continue
                q = deque([i]); comp = []
                seen.add(i)
                while q:
                    u = q.popleft()
                    comp.append(u)
                    for v in G[u]:
                        if v not in seen:
                            seen.add(v); q.append(v)
                comps.append(comp)

            # For each component, if any member is BT-pinned to row d_fix, then
            # the whole component must choose t[r,*,d_fix].
            forced_row = {}
            ok = True
            for comp in comps:
                pin_rows = { BT[r][i][0] for i in comp if i in BT[r] }  # set of pinned dest rows in this component
                if len(pin_rows) > 1:
                    raise RuntimeError(f"UNSAT precheck: round {r} a pair-component has conflicting BT rows {pin_rows}")
                if len(pin_rows) == 1:
                    d_fix = pin_rows.pop()
                    for i in comp:
                        forced_row[i] = d_fix

            # Count how many ions are allowed per dest row
            allow = [0]*n
            for i in ions:
                if i in BT[r]:
                    allow[BT[r][i][0]] += 1
                elif i in forced_row:
                    allow[forced_row[i]] += 1
                else:
                    # completely free w.r.t. row: can serve any row; count later
                    pass

            free_unforced = [i for i in ions if i not in BT[r] and i not in forced_row]
            # Each free-unforced ion can serve any row (subject to other constraints),
            # so the necessary condition is: for all d, allow[d] <= m
            bad = [(d, allow[d]) for d in range(n) if allow[d] > m]
            if bad:
                raise RuntimeError(f"UNSAT precheck: round {r} row over-subscribed by BT/pairs: {bad} (> m={m}).")

            # Also check that the **minimum required** per row can be met:
            # If every free-unforced ion is dragged to some rows by other logic, you’ll still fail,
            # but a necessary condition is sum_i q[r,d,c,i]=1 over c=0..m-1 must be satisfiable:
            # i.e., (# already forced to d) + (#free_unforced) >= m
            bad_need = [(d, allow[d]) for d in range(n) if allow[d] + len(free_unforced) < m]
            if bad_need:
                raise RuntimeError(
                    f"UNSAT precheck: round {r} row lacks candidates for (6b) exactness: "
                    f"{bad_need}; free_unforced={len(free_unforced)}, m={m}"
                )
        # -------------------------
        # Solve MaxSAT
        # -------------------------
        solver = RC2(wcnf)
        model = solver.compute()

        # -------------------------
        # If UNSAT (hard), rebuild hard-only with selectors and print core
        # -------------------------
        if model is None:
            core = CoreBuilder(vpool)

            # (0) a-cardinality per (r,k,j)
            for r in range(R):
                for krow in range(n):
                    for jcol in range(m):
                        lits = [var_a(r, krow, jcol, i) for i in ions]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        core.extend(f"(0).a_cell_(r={r},k={krow},j={jcol})", enc.clauses)

            # (1) initial: a[0,k,j, A_in[k,j]]
            for krow in range(n):
                for jcol in range(m):
                    ion = int(A_in[krow, jcol])
                    core.add(f"(1).init_(k={krow},j={jcol})", [var_a(0, krow, jcol, ion)])

            # (2) BT units
            for r in range(R):
                for ion in ions:
                    if is_reserved(r, ion):
                        d, c = BT[r][ion]
                        core.add(f"BT.x_(r={r},i={ion},c={c})", [var_x(r, ion, c)])
                        core.add(f"BT.t_(r={r},i={ion},d={d})", [var_t(r, ion, d)])
                        core.add(f"BT.w_(r={r},i={ion},b={c//k})", [var_w(r, ion, c//k)])
                        if r < R-1:
                            core.add(f"BT.a_next_(r={r},i={ion},d={d},c={c})", [var_a(r+1, d, c, ion)])

            # (3) y one-per (r,k,c)
            for r in range(R):
                for krow in range(n):
                    for c in range(m):
                        lits = [var_y(r, krow, c, i) for i in ions]
                        enc = CardEnc.atmost(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        core.extend(f"(3).y_one_per_(r={r},k={krow},c={c})", enc.clauses)

            # (3b) x onehot (free)
            for r in range(R):
                for ion in ions:
                    if not is_reserved(r, ion):
                        lits = [var_x(r, ion, c) for c in range(m)]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        core.extend(f"(3).x_onehot_(r={r},i={ion})", enc.clauses)

            # (4a) p row presence
            for r in range(R):
                for krow in range(n):
                    for ion in ions:
                        A_lits = [var_a(r, krow, jcol, ion) for jcol in range(m)]
                        core.add(f"(4a).p_row_presence_(r={r},k={krow},i={ion})", [-var_p(r, krow, ion)] + A_lits)
                        for aj in A_lits:
                            core.add(f"(4a).p_row_presence_(r={r},k={krow},i={ion})", [-aj, var_p(r, krow, ion)])

            # (4b) y <-> x & p
            for r in range(R):
                for krow in range(n):
                    for c in range(m):
                        for ion in ions:
                            yv = var_y(r, krow, c, ion)
                            pv = var_p(r, krow, ion)
                            if not is_reserved(r, ion):
                                xv = var_x(r, ion, c)
                                nm = f"(4b).y_link_(r={r},k={krow},c={c},i={ion})"
                                core.add(nm, [-yv, xv])
                                core.add(nm, [-yv, pv])
                                core.add(nm, [-xv, -pv, yv])
                            else:
                                d_fix, c_fix = BT[r][ion]
                                if c == c_fix:
                                    nm = f"(4b).y_link_res_eq_(r={r},k={krow},c={c},i={ion})"
                                    core.add(nm, [-yv, pv])
                                    core.add(nm, [-pv, yv])
                                else:
                                    nm = f"(4b).y_link_res_neq_(r={r},k={krow},c={c},i={ion})"
                                    core.add(nm, [-yv])

            # (5) t onehot (free)
            for r in range(R):
                for ion in ions:
                    if not is_reserved(r, ion):
                        lits = [var_t(r, ion, d) for d in range(n)]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        core.extend(f"(5).t_onehot_(r={r},i={ion})", enc.clauses)

            # (6) u per (r,c,d) equals-1 + implications
            for r in range(R):
                for c in range(m):
                    for d in range(n):
                        u_lits = [var_u(r, c, d, krow) for krow in range(n)]
                        enc = CardEnc.equals(lits=u_lits, encoding=EncType.ladder, vpool=vpool)
                        core.extend(f"(6).u_one_per_(r={r},c={c},d={d})", enc.clauses)
                        for krow in range(n):
                            uvar = var_u(r, c, d, krow)
                            ors_first, ors_second = [], []
                            for ion in ions:
                                if not is_reserved(r, ion):
                                    ors_first.append(var_y(r, krow, c, ion))
                                    ors_second.append(var_t(r, ion, d))
                            for ion in ions:
                                if is_reserved(r, ion):
                                    d_fix, c_fix = BT[r][ion]
                                    if d_fix == d and c_fix == c:
                                        ors_first.append(var_x(r, ion, c))
                                        ors_second.append(var_p(r, krow, ion))
                            nm = f"(6).u_implies_src_(r={r},c={c},d={d},k={krow})"
                            if not ors_first:
                                core.add(nm, [-uvar])
                            else:
                                core.add(nm, [-uvar] + ors_first)
                                core.add(nm, [-uvar] + ors_second)

            # (6b) q exactness
            for r in range(R):
                for d in range(n):
                    for c in range(m):
                        q_lits = []
                        for ion in ions:
                            qv = var_q(r, d, c, ion)
                            q_lits.append(qv)

                            if not is_reserved(r, ion):
                                xv = var_x(r, ion, c)
                                tv = var_t(r, ion, d)
                                nm = f"(6b).q_link_(r={r},d={d},c={c},i={ion})"
                                core.add(nm, [-qv, xv])
                                core.add(nm, [-qv, tv])
                                core.add(nm, [-xv, -tv, qv])
                            else:
                                d_fix, c_fix = BT[r][ion]
                                if d == d_fix and c == c_fix:
                                    core.add(f"(6b).q_unit_(r={r},d={d},c={c},i={ion})", [qv])
                                else:
                                    core.add(f"(6b).q_unit_(r={r},d={d},c={c},i={ion})", [-qv])
                        enc = CardEnc.equals(lits=q_lits, encoding=EncType.ladder, vpool=vpool)
                        core.extend(f"(6b).q_one_per_(r={r},d={d},c={c})", enc.clauses)

            # (7) w link (free)
            num_blocks = math.ceil(m / k)
            for r in range(R):
                for ion in ions:
                    if not is_reserved(r, ion):
                        for b in range(num_blocks):
                            cols = list(range(b * k, min((b + 1) * k, m)))
                            wv = var_w(r, ion, b)
                            nm = f"(7).w_link_(r={r},i={ion},b={b})"
                            core.add(nm, [-wv] + [var_x(r, ion, c) for c in cols])
                            for c in cols:
                                core.add(nm, [-var_x(r, ion, c), wv])

            # (8) pair equalities
            for r in range(R):
                for (i1, i2) in P_arr[r]:
                    for d in range(n):
                        nm = f"(8).pair_row_(r={r},{i1},{i2},d={d})"
                        core.add(nm, [-var_t(r, i1, d), var_t(r, i2, d)])
                        core.add(nm, [-var_t(r, i2, d), var_t(r, i1, d)])
                    for b in range(num_blocks):
                        nm = f"(8).pair_block_(r={r},{i1},{i2},b={b})"
                        core.add(nm, [-var_w(r, i1, b), var_w(r, i2, b)])
                        core.add(nm, [-var_w(r, i2, b), var_w(r, i1, b)])

            # (9) glue (non-reserved)
            for r in range(R - 1):
                for ion in ions:
                    if not is_reserved(r, ion):
                        for c in range(m):
                            for d in range(n):
                                av = var_a(r + 1, d, c, ion)
                                xv = var_x(r, ion, c)
                                tv = var_t(r, ion, d)
                                nm = f"(9).glue_(r={r},i={ion},c={c},d={d})"
                                core.add(nm, [-av, xv])
                                core.add(nm, [-av, tv])
                                core.add(nm, [-xv, -tv, av])

            names = core.build_and_solve()
            # If still SAT (shouldn't happen), just raise UNSAT as before
            if names is None:
                raise RuntimeError("UNSAT (hard) but core builder found SAT; check selector grouping.")
            # Print a concise UNSAT diagnosis and raise
            msg = ["UNSAT (hard). Minimal core groups:"]
            for nm in names:
                msg.append(f"  - {nm}")
            raise RuntimeError("\n".join(msg))

        # -------------------------
        # Decode SAT model
        # -------------------------
        model_set = set(l for l in model if l > 0)

        # Reconstruct A_1..A_R
        def get_lit(v): return (v in model_set)

        layouts: List[np.ndarray] = []
        cur = A_in.copy()
        for r in range(R):
            nxt = np.empty_like(cur)
            if r < R - 1:
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions:
                            if get_lit(var_a(r + 1, d, c, ion)):
                                found = ion; break
                        if found is None:
                            raise RuntimeError(f"could not reconstruct cell {(r+1, d, c)}")
                        nxt[d, c] = found
            else:
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions:
                            if get_lit(var_q(r, d, c, ion)):
                                found = ion; break
                        if found is None:
                            raise RuntimeError(f"could not reconstruct final cell {(r, d, c)}")
                        nxt[d, c] = found
            layouts.append(nxt)
            cur = nxt

        return layouts




    @staticmethod
    def _optimal_QMR_for_WISE2(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,
        wV: List[int] = None,
        wB_col: int = 1,
        wB_row: int = 1
    ) -> List[np.ndarray]:
        """"
            Inputs:
            - Initial Layout  A_in : nxm array of ion indexes
            - Circuit P_arr: list of size R, where for each round r: P_arr[r] contains the ion pairs for that round
            - Block size k
            - Boundary targets BT[r]: for round r, a map BT[r][i] = (d, c) saying ion i must target row d and column c as determined from the last SAT run

            Variables: 
            - x[r, i, c]    ion i has target column c in round r
            - t[r, i, d]    ion i has target row d in round r
            - w[r, i, b]    ion i is in block b from {0 ... m/k - 1} in round r
            - a[r, k, j, i] ion i is in cell (k, j) in round r
            - p[r, k, i]    ion i is present in row k in round r (for all ions, reserved or free)
            - y[r, k, c, i] ion i is in row k and has target column c in round r
            - h[r, δ]       horizontal band variable for round r; true if the horizontal displacement ≤ δ
            - v[r, δ]       vertical band variable for round r; true if the vertical displacement ≤ δ
            - u[r, c, d, k] helper: in round r, column c, destination row d is supplied by source row k

            Let:
            - Res_r  = { i | i ∈ BT[r] }             reserved ions for round r
            - Free_r = all ions \ Res_r              non-reserved ions for round r

            Hard constraints:

            (0) Exactly one ion per cell:
            - for all r, k, j : ∑_{i} a[r, k, j, i] = 1

            (1) Initial Layout:
            - for all k, j : a[0, k, j, A_in[k, j]]

            (2) Unit clauses fixing known targets (reserved ions):
            - for all r, i ∈ Res_r with BT[r][i] = (d, c):
                x[r, i, c]
                t[r, i, d]
                w[r, i, floor(c / k)]
                if r < R - 1: a[r+1, d, c, i]

            - for all r < R-1, i in Res_r with BT[r][i]=(k,j) and i in Res_(r+1) with BT[r+1][i]=(d,c)
                p[r+1,k,i]
                y[r+1,k,c,i]
                u[r+1,c,d,k]
                ** I am hoping this means for ions who live within a slice completely across all rounds have complete unit clauses so they have no contribution to the complexity**
                ** I recognise ions that are fixed in some rounds but not others will increase complexity but these should be small number***

            (3) Row–column mapping (horizontal permutation):
            - for all r, k, c : ∑_{i} y[r, k, c, i] = 1
            (every row k, every column c in round r is occupied by exactly one ion, regardless of whether it is reserved or free)
            - for all r, i ∉ Res_r : ∑_{c} x[r, i, c] = 1
            (only non-reserved ions choose their destination column; reserved ions already have x fixed in (2))

            (4) Row presence and linkage (for all ions):
            - for all r, k, i : p[r, k, i] ↔ (∨_{j} a[r, k, j, i])
            meaning p is true iff ion i actually sits in row k at the start of round r
            - for all r, k, c, i : y[r, k, c, i] ↔ (x[r, i, c] ∧ p[r, k, i])
            even for reserved ions, y is derived from their fixed x and their actual row p

            (5) Destination row assignment is unique:
            - for all r, i ∉ Res_r : ∑_{d} t[r, i, d] = 1
            (reserved ions already have t fixed in (2))

            (6) Column–destination row uniqueness (cheap version):
            - for all r, c, d, k :
                u[r, c, d, k] → (
                    ∨_{i ∉ Res_r} ( y[r, k, c, i] ∧ t[r, i, d] )
                    ∨
                    ∨_{i ∈ Res_r} ( x[r, i, c] ∧ t[r, i, d] ∧ p[r, k, i] )
                )
            (interpretation: column c, dest-row d is populated by exactly one source row k, either by a free ion that chose (c,d) or by a reserved ion that is actually in row k and was fixed to (d,c))
            - for all r, c, d : ∑_{k} u[r, c, d, k] = 1
            (6b) Destination-cell exactness:
            - for all r, d, c:
                ∑_{i} [ x[r, i, c] ∧ t[r, i, d] ] = 1
            (encode with a helper q[r,d,c,i] ↔ (x[r,i,c] ∧ t[r,i,d]))


            (7) Block membership:
            - for all r, i ∉ Res_r, b :
                w[r, i, b] ↔ ∨_{c ∈ {b*k, ..., (b+1)*k - 1}} x[r, i, c]
            (reserved ions already have w fixed in (2))

            (8) Pair constraints (same row and same block):
            - for all r : for all (i1, i2) in P_arr[r] :
                for all d : t[r, i1, d] ↔ t[r, i2, d]
                for all b : w[r, i1, b] ↔ w[r, i2, b]

            (9) Rounds glued together:
            - for all r < R-1, for all i ∉ Res_r, for all c, d :
                a[r+1, d, c, i] <-> ( x[r, i, c] ∧ t[r, i, d] )
            - for all r < R-1, for all i ∈ Res_r with BT[r][i] = (d, c) :
                a[r+1, d, c, i]

            (10) Band variable monotonicity:
            - for all r, δ < m - 1 : h[r, δ] → h[r, δ + 1]
            - for all r, δ < n - 1 : v[r, δ] → v[r, δ + 1]

            (11) Horizontal displacement linkage:
            - for all r, k, j, c, i ∉ Res_r :
                ( y[r, k, c, i] ∧ a[r, k, j, i] ) → h[r, |c - j|]
            (we only need to charge the band for movable ions; reserved ions’ horizontal move is already known)

            (12) Vertical displacement linkage:
            - for all r, k, d, i ∉ Res_r :
                ( t[r, i, d] ∧ p[r, k, i] ) → v[r, |d - k|]

            Soft constraints:
            - For each round r:
                * For δ in {0, ..., m-1}:
                    clause: ¬h[r, δ]  with weight w^H_δ
                * For δ in {0, ..., n-1}:
                    clause: ¬v[r, δ]  with weight w^V_δ
            - For each round r, for each (i1, i2) in P_arr[r]:
                * if i1 ∉ Res_r:
                    clause: ¬x[r, i1, m-1]  with weight w_B_col
                    clause: ¬t[r, i1, n-1]  with weight w_B_row
                * if i2 ∉ Res_r:
                    clause: ¬x[r, i2, m-1]  with weight w_B_col
                    clause: ¬t[r, i2, n-1]  with weight w_B_row

            Objective:
            - Minimize total weighted sum of broken soft clauses:
                Σ_r Σ_δ (w^H_δ * h[r, δ]) + Σ_r Σ_δ (w^V_δ * v[r, δ])
            which corresponds to minimizing the total number of horizontal and vertical odd–even passes across all rounds,
            while discouraging interacting ions from being placed on the outer boundary rows/columns unless necessary.

            Complexity (per SAT run / per slice):
            - Variables: O(R * n * m * (1 + n + m)) ≈ O(R * n² * m²) in the worst case but unit propagation will help this
            - Hard clauses: O(R * n² * m²)
            - Soft clauses: O(R * (n + m)) + O(total pairs)
        """
        
        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)

        # weights
        if wH is None:
            wH = [1 + d for d in range(m)]
        if wV is None:
            wV = [1 + d for d in range(n)]
        if BT is None:
            BT = [{} for _ in range(R)]


        # A_in rows
        row_of = {int(A_in[r,c]): r for r in range(A_in.shape[0]) for c in range(A_in.shape[1])}

        # 1) No duplicate destination cell in BT (conflicts with 6b)
        for r, bt in enumerate(BT):
            seen = {}
            for i, (d, c) in bt.items():
                if (d, c) in seen:
                    raise ValueError(f"UNSAT: BT[{r}] pins ions {seen[(d,c)]} and {i} to the same cell (d={d}, c={c}).")

        # 2) Pair vs BT conflicts (same round)
        for r, pairs in enumerate(P_arr):
            for i1, i2 in pairs:
                if i1 in BT[r] and i2 in BT[r]:
                    d1, c1 = BT[r][i1]; d2, c2 = BT[r][i2]
                    if d1 != d2:
                        raise ValueError(f"UNSAT: round {r} pair {(i1,i2)} BT rows differ: {d1} vs {d2}.")
                    if (c1 // k) != (c2 // k):
                        raise ValueError(f"UNSAT: round {r} pair {(i1,i2)} BT blocks differ: {c1}//{k} vs {c2}//{k}.")

        # 3) Round-0: same source row & same target column among reserved
        if len(BT) >= 1:
            buckets = {}
            for i, (d0, c0) in BT[0].items():
                sr = row_of[i]
                buckets.setdefault((sr, c0), []).append(i)
            bad = {key: vs for key, vs in buckets.items() if len(vs) > 1}
            if bad:
                raise ValueError(f"UNSAT: round 0 has reserved ions from the same start row targeting the same column: {bad}")
            
        # Pseudocode check for the y-collision you’re seeing
        collisions = {}
        r=0
        for i in A_in.flatten():
            if i in BT[r] and i in BT[r+1]:
                d1, c1 = BT[r][i]      # prev round dest row
                d2, c2 = BT[r+1][i]    # next round dest col/row
                key = (r+1, d1, c2)    # the (k,c) you unit-set y to 1
                collisions.setdefault(key, []).append(i)

        bad = {key: ids for key, ids in collisions.items() if len(ids) > 1}
        assert not bad, f"Reserved-prop y-units collide at {bad}"
        u_keys = {}
        for i in A_in.flatten():
            if i in BT[r] and i in BT[r+1]:
                d1, _ = BT[r][i]
                d2, c2 = BT[r+1][i]
                key = (r+1, c2, d2)   # destination cell (c2,d2), where you unit-set u[...] for k=d1
                u_keys.setdefault(key, set()).add(d1)

        bad_u = {key: ks for key, ks in u_keys.items() if len(ks) > 1}
        assert not bad_u, f"Reserved-prop u-units collide at {bad_u}"

        # Over-subscription of columns by BT in each round
        for r, bt in enumerate(BT):
            col_counts = {}
            for i, (_, c) in bt.items():
                col_counts[c] = col_counts.get(c, 0) + 1
            bad_cols = {c: cnt for c, cnt in col_counts.items() if cnt > n}
            if bad_cols:
                raise ValueError(f"UNSAT: BT[{r}] pins {bad_cols} ions to a single column, exceeds n={n}.")
        vpool = IDPool()

        # -------------------------
        # variable accessors
        # -------------------------
        def var_x(r, i, c):
            return vpool.id(('x', r, i, c))

        def var_t(r, i, d):
            return vpool.id(('t', r, i, d))

        def var_w(r, i, b):
            return vpool.id(('w', r, i, b))

        def var_a(r, krow, jcol, i):
            return vpool.id(('a', r, krow, jcol, i))

        def var_p(r, krow, i):
            return vpool.id(('p', r, krow, i))

        def var_y(r, krow, c, i):
            return vpool.id(('y', r, krow, c, i))

        def var_h(r, delta):
            return vpool.id(('h', r, delta))

        def var_v(r, delta):
            return vpool.id(('v', r, delta))

        def var_u(r, c, d, krow):
            return vpool.id(('u', r, c, d, krow))

        # NEW for 6b
        def var_q(r, d, c, i):
            return vpool.id(('q', r, d, c, i))

        ions: List[int] = sorted(int(x) for x in A_in.flatten())

        wcnf = WCNF()

        # (0) exactly one ion per cell for rounds 0..R-1 (like your code)
        for r in range(R):
            for krow in range(n):
                for jcol in range(m):
                    lits = [var_a(r, krow, jcol, i) for i in ions]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)

        # (1) initial layout
        for krow in range(n):
            for jcol in range(m):
                ion = int(A_in[krow, jcol])
                wcnf.append([var_a(0, krow, jcol, ion)])

        def is_reserved(r, ion):
            return ion in BT[r]

        # (2) unit clauses for reserved ions
        for r in range(R):
            for ion in ions:
                if is_reserved(r, ion):
                    d, c = BT[r][ion]
                    wcnf.append([var_x(r, ion, c)])
                    wcnf.append([var_t(r, ion, d)])
                    b = c // k
                    wcnf.append([var_w(r, ion, b)])
                    if r < R - 1:
                        wcnf.append([var_a(r + 1, d, c, ion)])

        # # NEW reserved-propagation
        # for r in range(R - 1):
        #     for ion in ions:
        #         if is_reserved(r, ion) and is_reserved(r + 1, ion):
        #             d1, c1 = BT[r][ion]
        #             d2, c2 = BT[r + 1][ion]
        #             wcnf.append([var_p(r + 1, d1, ion)])
        #             wcnf.append([var_y(r + 1, d1, c2, ion)])
        #             wcnf.append([var_u(r + 1, c2, d2, d1)])

        # (3) row–column mapping
        for r in range(R):
            # each (row, col) has one ion
            for krow in range(n):
                for c in range(m):
                    lits = [var_y(r, krow, c, i) for i in ions]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)
            # free ions pick one column
            for ion in ions:
                if not is_reserved(r, ion):
                    lits = [var_x(r, ion, c) for c in range(m)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)

        # (4) row presence + y linkage
        for r in range(R):
            # p[r,k,i] <-> OR_j a[r,k,j,i]
            for krow in range(n):
                for ion in ions:
                    A_lits = [var_a(r, krow, jcol, ion) for jcol in range(m)]
                    wcnf.append([-var_p(r, krow, ion)] + A_lits)
                    for aj in A_lits:
                        wcnf.append([-aj, var_p(r, krow, ion)])

            # y[r,k,c,i] <-> x[r,i,c] & p[r,k,i]
            for krow in range(n):
                for c in range(m):
                    for ion in ions:
                        yv = var_y(r, krow, c, ion)
                        xv = var_x(r, ion, c) if not is_reserved(r, ion) else var_x(r, ion, BT[r][ion][1])
                        pv = var_p(r, krow, ion)
                        wcnf.append([-yv, xv])
                        wcnf.append([-yv, pv])
                        wcnf.append([-xv, -pv, yv])

        # (5) destination-row unique (free)
        for r in range(R):
            for ion in ions:
                if not is_reserved(r, ion):
                    lits = [var_t(r, ion, d) for d in range(n)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)

        # (6) column–dest-row uniqueness (cheap)
        for r in range(R):
            for c in range(m):
                for d in range(n):
                    u_lits = [var_u(r, c, d, krow) for krow in range(n)]
                    enc = CardEnc.equals(lits=u_lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)

                    for krow in range(n):
                        uvar = var_u(r, c, d, krow)
                        ors_first = []
                        ors_second = []

                        # free candidates
                        for ion in ions:
                            if not is_reserved(r, ion):
                                yv = var_y(r, krow, c, ion)
                                tv = var_t(r, ion, d)
                                ors_first.append(yv)
                                ors_second.append(tv)

                        # reserved candidates
                        for ion in ions:
                            if is_reserved(r, ion):
                                d_fix, c_fix = BT[r][ion]
                                if d_fix == d and c_fix == c:
                                    xv = var_x(r, ion, c)
                                    pv = var_p(r, krow, ion)
                                    ors_first.append(xv)
                                    ors_second.append(pv)

                        if not ors_first:
                            wcnf.append([-uvar])
                        else:
                            wcnf.append([-uvar] + ors_first)
                            wcnf.append([-uvar] + ors_second)

        # (6b) destination-cell exactness: for all r,d,c: sum_i q[r,d,c,i] = 1, and q <-> (x & t)
        for r in range(R):
            for d in range(n):
                for c in range(m):
                    q_lits = []
                    for ion in ions:
                        qv = var_q(r, d, c, ion)
                        q_lits.append(qv)
                        xv = var_x(r, ion, c) if not is_reserved(r, ion) else var_x(r, ion, BT[r][ion][1])
                        tv = var_t(r, ion, d) if not is_reserved(r, ion) else var_t(r, ion, BT[r][ion][0])
                        # q -> x, q -> t
                        wcnf.append([-qv, xv])
                        wcnf.append([-qv, tv])
                        # (x & t) -> q
                        wcnf.append([-xv, -tv, qv])
                    enc = CardEnc.equals(lits=q_lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        wcnf.append(cl)

        # (7) block membership
        num_blocks = math.ceil(m / k)
        for r in range(R):
            for ion in ions:
                if not is_reserved(r, ion):
                    for b in range(num_blocks):
                        cols = list(range(b * k, min((b + 1) * k, m)))
                        lits = [var_x(r, ion, c) for c in cols]
                        wv = var_w(r, ion, b)
                        wcnf.append([-wv] + lits)
                        for lc in lits:
                            wcnf.append([-lc, wv])

        # (8) pair constraints
        for r in range(R):
            for (i1, i2) in P_arr[r]:
                for d in range(n):
                    wcnf.append([-var_t(r, i1, d), var_t(r, i2, d)])
                    wcnf.append([-var_t(r, i2, d), var_t(r, i1, d)])
                for b in range(num_blocks):
                    wcnf.append([-var_w(r, i1, b), var_w(r, i2, b)])
                    wcnf.append([-var_w(r, i2, b), var_w(r, i1, b)])

        # (9) rounds glued together
        for r in range(R - 1):
            for ion in ions:
                if not is_reserved(r, ion):
                    for c in range(m):
                        for d in range(n):
                            av = var_a(r + 1, d, c, ion)
                            xv = var_x(r, ion, c)
                            tv = var_t(r, ion, d)
                            wcnf.append([-av, xv])
                            wcnf.append([-av, tv])
                            wcnf.append([-xv, -tv, av])

        # (10) band monotonicity
        for r in range(R):
            for delta in range(m - 1):
                wcnf.append([-var_h(r, delta), var_h(r, delta + 1)])
            for delta in range(n - 1):
                wcnf.append([-var_v(r, delta), var_v(r, delta + 1)])

        # (11) horizontal linkage
        for r in range(R):
            for krow in range(n):
                for jcol in range(m):
                    for c in range(m):
                        dist = abs(c - jcol)
                        for ion in ions:
                            if not is_reserved(r, ion):
                                yv = var_y(r, krow, c, ion)
                                av = var_a(r, krow, jcol, ion)
                                hv = var_h(r, dist)
                                wcnf.append([-yv, -av, hv])

        # (12) vertical linkage
        for r in range(R):
            for krow in range(n):
                for d in range(n):
                    dist = abs(d - krow)
                    for ion in ions:
                        if not is_reserved(r, ion):
                            tv = var_t(r, ion, d)
                            pv = var_p(r, krow, ion)
                            vv = var_v(r, dist)
                            wcnf.append([-tv, -pv, vv])

        # Soft constraints
        for r in range(R):
            for delta in range(m):
                wcnf.append([-var_h(r, delta)], weight=wH[delta])
            for delta in range(n):
                wcnf.append([-var_v(r, delta)], weight=wV[delta])
            for (i1, i2) in P_arr[r]:
                if not is_reserved(r, i1):
                    wcnf.append([-var_x(r, i1, m - 1)], weight=wB_col)
                    wcnf.append([-var_t(r, i1, n - 1)], weight=wB_row)
                if not is_reserved(r, i2):
                    wcnf.append([-var_x(r, i2, m - 1)], weight=wB_col)
                    wcnf.append([-var_t(r, i2, n - 1)], weight=wB_row)

        # Solve
        solver = RC2(wcnf)
        model = solver.compute()
        if model is None:
            raise RuntimeError("UNSAT")

        model_set = set(l for l in model if l > 0)

        # Reconstruct A_1..A_R
        layouts: List[np.ndarray] = []
        cur = A_in.copy()
        for r in range(R):
            nxt = np.empty_like(cur)
            if r < R - 1:
                # use a[r+1,...]
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions:
                            v = var_a(r + 1, d, c, ion)
                            if v in model_set:
                                found = ion
                                break
                        if found is None:
                            raise RuntimeError(f"could not reconstruct cell {(r+1, d, c)}")
                        nxt[d, c] = found
            else:
                # last round: use q[r,d,c,i] from 6b
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions:
                            qv = var_q(r, d, c, ion)
                            if qv in model_set:
                                found = ion
                                break
                        # if found is None:
                        #     # final fallback to x&t
                        #     for ion in ions:
                        #         if var_x(r, ion, c) in model_set and var_t(r, ion, d) in model_set:
                        #             found = ion
                        #             break
                        if found is None:
                            raise RuntimeError(f"could not reconstruct final cell {(r, d, c)}")
                        nxt[d, c] = found
            layouts.append(nxt)
            cur = nxt

        return layouts


    @staticmethod
    def _optimal_phaseB_min_passes_sat(A_in, T_in):
        """
        Exact Phase-B optimizer (WISE) via SAT + binary search on P (passes).
        Minimizes the TRUE objective: min # odd–even passes = min max_r max |j - c|.
        Enforces: (i) row-wise permutations only, (ii) in each global column c, each destination row appears exactly once.


        Returns:
            P_opt (int), desired (np.ndarray int [n x m])
        """
        A = np.asarray(A_in, dtype=int)
        T = np.asarray(T_in, dtype=int)
        n, m = A.shape

        # dest row per ion id
        ion_to_dest = {}
        for r in range(n):
            for c in range(m):
                ion_to_dest[int(T[r, c])] = r

        def solve_P(P):

            # dest row per ion
            ion_to_dest = {}
            for r in range(n):
                for c in range(m):
                    ion_to_dest[int(T[r, c])] = r

            # ONE shared var pool for all variables (decision + aux from CardEnc)
            vpool = IDPool(start_from=1)

            def var_x(r, j, c):  # ion currently at (r,j) goes to column c (stays in row r)
                return vpool.id(('x', r, j, c))

            cnf = CNF()

            # (1a) For each (r,j): exactly one c with |j-c| <= P
            for r in range(n):
                for j in range(m):
                    cols = [var_x(r, j, c) for c in range(max(0, j - P), min(m, j + P + 1))]
                    if not cols:
                        return False, None
                    # equals-1 = atleast-1 + atmost-1; share the SAME vpool
                    cnf.extend(CardEnc.equals(lits=cols, encoding=EncType.ladder, vpool=vpool))

            # (1b) For each (r,c): exactly one j with |j-c| <= P
            for r in range(n):
                for c in range(m):
                    jL, jR = max(0, c - P), min(m, c + P + 1)
                    lits = [var_x(r, j, c) for j in range(jL, jR)]
                    if not lits:
                        return False, None
                    cnf.extend(CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool))

            # (2) For each (c,d): at most one ion with dest d lands in column c
            for c in range(m):
                for d in range(n):
                    lits = []
                    for r in range(n):
                        for j in range(m):
                            if abs(j - c) <= P and ion_to_dest[int(A[r, j])] == d:
                                lits.append(var_x(r, j, c))
                    if not lits:
                        # For P = m-1, this should never happen if A,T are consistent
                        return False, None
                    cnf.extend(CardEnc.atmost(lits=lits, bound=1, encoding=EncType.pairwise, vpool=vpool))

            # Solve
            with Minisat22(bootstrap_with=cnf.clauses) as sat:
                if not sat.solve():
                    return False, None
                model = set(l for l in sat.get_model() if l > 0)

            # Extract layout
            desired = -np.ones((n, m), dtype=int)
            for r in range(n):
                for j in range(m):
                    for c in range(max(0, j - P), min(m, j + P + 1)):
                        if vpool.id(('x', r, j, c)) in model:
                            desired[r, c] = int(A[r, j])
                            break

            return True, desired

        # Binary search on P
        lo, hi = 0, m-1
        best_P, best_desired = None, None
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, des = solve_P(mid)
            if ok:
                best_P, best_desired = mid, des
                hi = mid - 1
            else:
                lo = mid + 1

        if best_desired is None:
            # Helpful diagnostics for P = m-1 (no bandlimit): check global feasibility quickly
            # (it should always exist by the edge-coloring argument; if not, inputs are inconsistent)
            raise RuntimeError("No feasible Phase-B assignment found; check ion_to_dest and inputs.")
        return best_P, best_desired
        
    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: QCCDWiseArch,
        arrangement: Mapping[Trap, Sequence[Ion]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
    ) -> Tuple[Mapping[int, float], float]:
        """
        Shapes:
        rows = wiseArch.n, cols = wiseArch.m, stride k = wiseArch.k
        Arrays hold ion IDs (ints).
        Logs:
        "Parrellel split"
        "ROWSWAP {rowIdx} {ionIdx1} {ionIdx2}"
        "COLSWAP {colIdx} {ionIdx1} {ionIdx2}"
        "Parrellel row reconfig"
        """
        heatingRates: Mapping[int, float]  = {}
        for _, ions in arrangement.items():
            for ion in ions:
                heatingRates[ion.idx] = 0.0
        timeElapsed = 0.0

        row_swap_time = Move.MOVING_TIME+Merge.MERGING_TIME+CrystalRotation.ROTATION_TIME+Split.SPLITTING_TIME+Move.MOVING_TIME
        row_swap_heating = Move.MOVING_TIME*Move.HEATING_RATE+Merge.MERGING_TIME*Merge.HEATING_RATE+CrystalRotation.ROTATION_TIME*CrystalRotation.HEATING_RATE+Split.SPLITTING_TIME*Split.HEATING_RATE+Move.MOVING_TIME*Move.HEATING_RATE
        col_swap_time = (2*JunctionCrossing.CROSSING_TIME)+(4*JunctionCrossing.CROSSING_TIME+Move.MOVING_TIME)*2
        col_swap_heating_rate = (6*JunctionCrossing.CROSSING_TIME*JunctionCrossing.HEATING_RATE)+Move.MOVING_TIME*Move.HEATING_RATE

        n = wiseArch.n  # rows
        m = wiseArch.m*wiseArch.k  # cols
        k = wiseArch.k  # column stride for junction batching

        A = np.array(oldAssignment, dtype=int)      # current
        T = np.array(newAssignment, dtype=int)      # target
  

        # ---------- helper: odd-even passes ----------
        def row_pass_by_rank(even_phase: bool, row_rank: List[Dict[int,int]]) -> bool:
            maxSwapsInRow=0
            start = 0 if even_phase else 1
            for r in range(n):
                swapsInRow=0
                rank = row_rank[r]
                for c in range(start, m-1, 2):
                    a = int(A[r, c]); b = int(A[r, c+1])
                    if rank[a] > rank[b]:
                        A[r, c], A[r, c+1] = b, a
                        #ROWSWAP {r} {a} {b}
                        heatingRates[a]+=row_swap_heating
                        heatingRates[b]+=row_swap_heating
                        swapsInRow+=1
                if swapsInRow > maxSwapsInRow:
                    maxSwapsInRow = swapsInRow
            return maxSwapsInRow>0

        def col_bucket_pass(even_phase: bool, bucket_mod: int, ion_to_dest_row: Dict[int,int]) -> bool:
            maxSwapsInCol=0
            start = 0 if even_phase else 1
            for c in range(bucket_mod, m, k):
                swapsInCol=0
                for r in range(start, n-1, 2):
                    a = int(A[r, c]); b = int(A[r+1, c])
                    if ion_to_dest_row[a] > ion_to_dest_row[b]:
                        A[r, c], A[r+1, c] = b, a
                        #"COLSWAP {c} {a} {b}"
                        heatingRates[a]+=col_swap_heating_rate
                        heatingRates[b]+=col_swap_heating_rate
                        swapsInCol+=1
                if swapsInCol > maxSwapsInCol:
                    maxSwapsInCol = swapsInCol
            return maxSwapsInCol>0

        # ---------- destination row map (ion -> dest row) ----------
        ion_to_dest_row: Dict[int,int] = {}
        ion_to_dest_col: Dict[int, int] = {}
        for r in range(n):
            for c in range(m):
                ion_to_dest_row[int(T[r, c])] = r
                ion_to_dest_col[int(T[r,c])]=c

        # =========================
        # Phase A: parallel split
        # =========================
        timeElapsed += Split.SPLITTING_TIME
        for idx in heatingRates.keys():
            heatingRates[idx]+=Split.HEATING_RATE*Split.SPLITTING_TIME

        # ==========================================================
        # Phase B: ensure each column has unique destination rows
        # via m perfect matchings (edge-coloring); then 1D odd-even
        # per row to realize the assigned per-row permutation.
        # ==================
        # sanity = GlobalReconfigurations.sanity(A, T)
        # est_cost, desired_row_order = GlobalReconfigurations._optimal_phaseB_min_passes_sat_heavy(A, T)
        

        desired_row_order = np.zeros_like(A)
        for r in range(n):
            for ionidx in A[r]:
                desired_row_order[r][ion_to_dest_col[ionidx]]=ionidx

        dest_rows = [[None]*m for _ in range(n)]
        for r in range(n):
            for c in range(m):
                dest_rows[r][c] = ion_to_dest_row[desired_row_order[r][c]]

        for c in range(m):
            assert (len(set(dest_rows[r][c] for r in range(n)))==n)

        for r in range(n):
            assert len(set(A[r]).difference(set(desired_row_order[r])))==0
        # Row ranks for Phase B permutation
        row_rank_phaseB: List[Dict[int,int]] = []
        for r in range(n):
            row_rank_phaseB.append({ion: idx for idx, ion in enumerate(desired_row_order[r])})

        # Execute ≤ m odd–even steps to realize the permutation per row
        acc_cost = 0
        for _ in range(m):
            oddpass = row_pass_by_rank(True,  row_rank_phaseB)
            evenpass = row_pass_by_rank(False, row_rank_phaseB)
            timeElapsed+=oddpass*row_swap_time
            timeElapsed+=evenpass*row_swap_time
            acc_cost+=int(oddpass)+ int(evenpass)

        # diff = est_cost-acc_cost

        # ==========================================================
        # Phase C: vertical odd–even with k-way parallel buckets,
        # comparator = destination row (time-optimal ≤ n).
        # ==========================================================
        for t in range(k):
            for _ in range(n):
                timeElapsed+=col_bucket_pass(True,  t, ion_to_dest_row)*col_swap_time
                timeElapsed+=col_bucket_pass(False, t, ion_to_dest_row)*col_swap_time
            if t < k - 1:
                #"Parrellel row reconfig"
                timeElapsed += k*row_swap_time
                for idx in heatingRates.keys():
                    heatingRates[idx]+=row_swap_heating

        # ==========================================================
        # Phase D: final row-wise odd–even to exact target order.
        # ==========================================================
        row_rank_final: List[Dict[int,int]] = []
        for r in range(n):
            row_rank_final.append({ion: idx for idx, ion in enumerate(T[r, :])})
        for _ in range(m):
            timeElapsed+=row_pass_by_rank(True,  row_rank_final)*row_swap_time
            timeElapsed+=row_pass_by_rank(False, row_rank_final)*row_swap_time

        return heatingRates, timeElapsed



class Split(CrystalOperation):
    KEY = Operations.SPLIT
    SPLITTING_TIME = 80e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], None],
        involvedComponents: Sequence[QCCDComponent],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.SPLITTING_TIME

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is not None:
            return False
        if len(self._trap.ions) == 0:
            return False
        if self._crossing.getEdgeIon(self._trap) != self._ion:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Split: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is not None:
            raise ValueError(
                f"Split: crossing is already occupied by ion {self._crossing.ion.idx}"
            )
        if len(self._trap.ions) == 0:
            raise ValueError(f"Split: trap {self._trap.idx} has no ions")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.getEdgeIon(trap)
            trap.removeIon(ion)
            crossing.setIon(ion, trap)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.SPLITTING_TIME)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.SPLITTING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            trap=trap,
            crossing=crossing,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class Merge(CrystalOperation):
    KEY = Operations.MERGE
    MERGING_TIME = 80e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MERGING_TIME

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.getEdgeIon(self._trap)]
        self._addOns = f" {self._crossing.getEdgeIon(self._trap).label}"

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasTrap(self._trap):
            return False
        if self._crossing.ion is None:
            return False
        if self._crossing.ion != self._ion:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.hasTrap(self._trap):
            raise ValueError(f"Merge: crossing does not include trap {self._trap.idx}")
        if self._crossing.ion is None:
            raise ValueError(f"Merge: crossing is empty")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            ion = crossing.ion
            crossing.clearIon()
            edge_ion = crossing.getEdgeIon(trap) if trap.ions else None
            idx = trap.ions.index(edge_ion) if trap.ions else 0
            if len(trap.ions)==1:
                offset = 1 if ion.pos[0]-edge_ion.pos[0]+ion.pos[1]-edge_ion.pos[1]>0 else 0
                adjacentIon=None
            else:
                offset=idx>0
                adjacentIon=edge_ion
            trap.addIon(ion, adjacentIon=adjacentIon, offset=offset)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.MERGING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            trap=trap,
            involvedComponents=[trap, crossing, *crossing.connection],
        )


class CrystalRotation(CrystalOperation):
    KEY = Operations.CRYSTAL_ROTATION
    ROTATION_TIME = (
        42e-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )
    HEATING_RATE = (
        0.3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        trap: Trap,
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._trap: Trap = trap

    def calculateOperationTime(self) -> None:
        self._operationTime = self.ROTATION_TIME

    @property
    def isApplicable(self) -> bool:
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, trap: Trap):
        def run():
            ions = list(trap.ions).copy()[::-1]
            for ion in ions:
                trap.removeIon(ion)
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.ROTATION_TIME)

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )



class CoolingOperation(CrystalOperation):
    KEY = Operations.RECOOLING
    COOLING_TIME = 400-6  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    HEATING_RATE = (
        0.1  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
    
    def calculateOperationTime(self) -> None:
        self._operationTime =  self.COOLING_TIME

    @property
    def isApplicable(self) -> bool:
        if not self._trap.hasCoolingIon:
            return False
        return super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._trap.hasCoolingIon:
            raise ValueError(f"CoolingOperation: trap {self._trap.idx} does not include a cooling ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, trap: Trap
    ):
        def run():
            trap.coolTrap()
            trap.addMotionalEnergy(cls.HEATING_RATE * cls.COOLING_TIME)

        return cls(
            run=lambda _: run(),
            trap=trap,
            involvedComponents=[trap],
        )




class Move(Operation):
    KEY = Operations.MOVE
    MOVING_TIME = 5e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        0.1  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.MOVING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        self._dephasingFidelity= 1 # little to no idling due to shuttling being fast

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._crossing.ion]
        self._addOns = f" {self._crossing.ion.label}"

    @property
    def isApplicable(self) -> bool:
        return bool(self._crossing.ion) and self._ion == self._crossing.ion and super().isApplicable

    def _checkApplicability(self) -> None:
        if not self._crossing.ion:
            raise ValueError(f"Move: crossing does not contain ion")
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(cls, crossing: Crossing, ion: Optional[Ion] = None):
        def run():
            crossing.ion.addMotionalEnergy(cls.HEATING_RATE * cls.MOVING_TIME)
            crossing.moveIon()

        return cls(
            run=lambda _: run(),
            ion=ion,
            crossing=crossing,
            involvedComponents=[crossing],
        )

# TODO: junction crossing should really go over the junction to the next crossing
class JunctionCrossing(Operation):
    KEY = Operations.JUNCTION_CROSSING
    CROSSING_TIME = 50e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.CROSSING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        # FIXME might be inaccurate
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/2.2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == self._junction.DEFAULT_CAPACITY:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) > 0:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.CROSSING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )






class PhysicalCrossingSwap(Operation):
    KEY = Operations.JUNCTION_CROSSING
    CROSSING_TIME = 100e-6  # TABLE I https://arxiv.org/pdf/2004.04706
    HEATING_RATE = (
        3  # TABLE IV https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
    )

    def __init__(
        self,
        run: Callable[[Any], bool],
        involvedComponents: Sequence["QCCDComponent"],
        **kwargs,
    ) -> None:
        super().__init__(run, involvedComponents=involvedComponents, **kwargs)
        self._crossing: Crossing = kwargs["crossing"]
        self._junction: Junction = kwargs["junction"]
        self._ion: Ion = kwargs["ion"]

    def calculateOperationTime(self) -> None:
        self._operationTime = self.CROSSING_TIME

    def calculateFidelity(self) -> None:
        self._fidelity = 1  # NOISE INCORPORATED INTO HEATING MODEL

    def calculateDephasingFidelity(self) -> None:
        # FIXME might be inaccurate
        self.calculateOperationTime()
        self._dephasingFidelity = 1 - (1-np.exp(-self.operationTime()/2.2))/2  # Dephasing noise https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330

    def _generateLabelAddOns(self) -> None:
        self._involvedIonsForLabel = [self._ion] if self._ion else []
        self._addOns = f" {self._ion.label}" if self._ion else ""

    @property
    def isApplicable(self) -> bool:
        if not self._crossing.hasJunction(self._junction):
            return False
        if not self._crossing.ion and len(self._junction.ions) == 0:
            return False
        if self._crossing.ion and self._crossing.ion != self._ion:
            return False
        if self._junction.ions and self._junction.ions[0] != self._ion:
            return False
        if self._crossing.ion and len(self._junction.ions) == self._junction.DEFAULT_CAPACITY:
            return False
        return super().isApplicable
    
    def _checkApplicability(self) -> None:
        if not self._crossing.hasJunction(self._junction):
            raise ValueError(
                f"JunctionCrossing: crossing does not contain junction {self._junction.idx}"
            )
        if not self._crossing.ion and len(self._junction.ions) == 0:
            raise ValueError(
                f"JunctionCrossing: neither junction nor crossing has an ion"
            )
        return super()._checkApplicability()

    @classmethod
    def physicalOperation(
        cls, junction: Junction, crossing: Crossing, ion: Optional[Ion] = None
    ):
        def run():
            if not crossing.ion and len(junction.ions) > 0:
                ion = junction.ions[0]
                crossing.setIon(ion, junction)
                junction.removeIon(ion)
            else:
                ion = crossing.ion
                crossing.clearIon()
                junction.addIon(ion)
            ion.addMotionalEnergy(cls.HEATING_RATE * cls.CROSSING_TIME)

        return cls(
            run=lambda _: run(),
            ion=ion,
            junction=junction,
            crossing=crossing,
            involvedComponents=[junction, crossing],
        )


class ParallelOperation(Operation):
    KEY = Operations.PARALLEL

    def __init__(
        self, run: Callable[[Any], bool], operations: Sequence[Operation], **kwargs
    ) -> None:
        super().__init__(run, **kwargs, operations=operations)
        self._operations = operations

    def calculateOperationTime(self) -> None:
        for op in self._operations:
            op.calculateOperationTime()
        self._operationTime = max(op.operationTime() for op in self._operations)

    def calculateDephasingFidelity(self) -> None:
        for op in self._operations:
            op.calculateDephasingFidelity()
        self._dephasingFidelity = float(max([op.dephasingFidelity() for op in self._operations]))


    def calculateFidelity(self) -> None:
        for op in self._operations:
            op.calculateFidelity()
        # assuming independence between parallel operations
        self._fidelity = float(np.prod([op.fidelity() for op in self._operations]))

    def _generateLabelAddOns(self) -> None:
        self._addOns = ""
        for op in self._operations:
            self._addOns += f" {op.KEY.name}"

    @property
    def isApplicable(self) -> bool:
        return all(op.isApplicable for op in self.operations)
    
    def _checkApplicability(self) -> None:
        return True

    @property
    def operations(self) -> Sequence[Operation]:
        return self._operations

    @classmethod
    def physicalOperation(cls, operationsToStart: Sequence[Operation], operationsStarted: Sequence[Operation]):
        def run():
            for op in np.random.permutation(operationsToStart):
                op.run()

        involvedComponents = []
        operations = list(operationsStarted)+list(operationsToStart)
        for op in operations:
            involvedComponents += list(op.involvedComponents)
        return cls(
            run=lambda _: run(),
            operations=operations,
            involvedComponents=set(involvedComponents),
        )
    
