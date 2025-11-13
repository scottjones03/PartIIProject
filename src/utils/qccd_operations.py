import numpy as np
from typing import (
    Sequence,
    List,
    Optional,
    Callable,
    Any,
    Mapping,
    Set
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
from multiprocessing import Process, Pipe
from collections import Counter
import signal
from pysat.formula import IDPool, WCNF, CNF
from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22  
from pysat.examples.rc2 import RC2
import time
import pickle 
import os
import tempfile
import multiprocessing as mp

def _sat_worker_from_file(cnf_path: str, result_path: str):
    """
    Worker process entry for plain SAT:
      - loads CNF from cnf_path,
      - runs Minisat22,
      - dumps {'sat': bool, 'model': list[int] or None, 'time': elapsed}
        or {'error': repr(e)} to result_path via pickle.
    """
    try:
        t0 = time.time()
        with open(cnf_path, "rb") as f:
            cnf = pickle.load(f)   # CNF object or just clauses

        with Minisat22(bootstrap_with=cnf.clauses) as sat:
            sat_ok = sat.solve()
            model = sat.get_model() if sat_ok else None

        t1 = time.time()
        data = {
            "sat": bool(sat_ok),
            "model": model,
            "time": t1 - t0,
        }

    except KeyboardInterrupt as e:
        data = {
            "error": f"KeyboardInterrupt in SAT worker: {repr(e)}",
        }

    except Exception as e:
        data = {
            "error": repr(e),
        }

    try:
        with open(result_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass
def run_sat_with_timeout_file(
    cnf: CNF,
    timeout_s: float,
    debug_prefix: str = "[WISE]",
):
    """
    Run Minisat22 on 'cnf' in a separate process with a wall-clock timeout.

    Returns: (sat_ok, model, status)
      - status == "ok"        : sat_ok is True/False, model is list[int] or None.
      - status == "timeout"   : sat_ok, model are None (worker killed by timeout).
      - status == "error"     : sat_ok, model are None (worker crashed).
      - status == "user_abort": sat_ok, model are None (parent got KeyboardInterrupt
                                while waiting; worker terminated & cleaned up).
    """

    if timeout_s is not None and timeout_s <= 0:
        if debug_prefix:
            print(
                f"{debug_prefix} SAT disabled (timeout <= 0); treating as timeout.",
                flush=True,
            )
        return None, None, "timeout"

    with tempfile.TemporaryDirectory() as tmpdir:
        cnf_path = os.path.join(tmpdir, "instance.cnf.pkl")
        result_path = os.path.join(tmpdir, "result_sat.pkl")

        with open(cnf_path, "wb") as f:
            pickle.dump(cnf, f)

        p = mp.Process(target=_sat_worker_from_file,
                       args=(cnf_path, result_path))
        p.daemon = True

        if debug_prefix:
            print(
                f"{debug_prefix} SAT worker starting (timeout={timeout_s:.1f}s) "
                f"for CNF: clauses={len(cnf.clauses)}",
                flush=True,
            )

        try:
            p.start()
        except KeyboardInterrupt:
            if debug_prefix:
                print(
                    f"{debug_prefix} SAT start interrupted by user; "
                    "terminating worker and returning user_abort.",
                    flush=True,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return None, None, "user_abort"

        start = time.time()
        status = None

        try:
            while True:
                p.join(0.5)

                if not p.is_alive():
                    status = "finished"
                    break

                elapsed = time.time() - start
                if elapsed >= timeout_s:
                    status = "timeout"
                    break

        except KeyboardInterrupt:
            if debug_prefix:
                elapsed = time.time() - start
                print(
                    f"{debug_prefix} SAT join interrupted by user after "
                    f"{elapsed:.3f}s; terminating worker (user_abort).",
                    flush=True,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return None, None, "user_abort"

        if status == "timeout":
            if debug_prefix:
                elapsed = time.time() - start
                print(
                    f"{debug_prefix} SAT worker exceeded {timeout_s:.1f}s "
                    f"(elapsed={elapsed:.3f}s); terminating (timeout).",
                    flush=True,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return None, None, "timeout"

        if not os.path.exists(result_path):
            if debug_prefix:
                print(
                    f"{debug_prefix} SAT worker finished but produced no result file; "
                    "treating as error.",
                    flush=True,
                )
            return None, None, "error"

        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            if debug_prefix:
                print(
                    f"{debug_prefix} SAT worker result read error: {e!r}; "
                    "treating as error.",
                    flush=True,
                )
            return None, None, "error"

        if "error" in data:
            if debug_prefix:
                print(
                    f"{debug_prefix} SAT worker raised: {data['error']}; "
                    "treating as error.",
                    flush=True,
                )
            return None, None, "error"

        if debug_prefix:
            print(
                f"{debug_prefix} SAT worker finished in {data.get('time', 0.0):.3f}s, "
                f"SAT={data.get('sat')}",
                flush=True,
            )

        return bool(data.get("sat")), data.get("model"), "ok"


def _rc2_worker_from_file(wcnf_path: str, result_path: str):
    """
    Worker process entry:
      - loads WCNF from wcnf_path,
      - runs RC2,
      - dumps {'model': model, 'cost': cost, 'time': elapsed}
        or {'error': repr(e)} to result_path via pickle.
    """
    try:
        t0 = time.time()
        with open(wcnf_path, "rb") as f:
            wcnf = pickle.load(f)

        rc2 = RC2(wcnf)
        model = rc2.compute()
        cost = rc2.cost if model is not None else None
        t1 = time.time()

        data = {
            "model": model,
            "cost": cost,
            "time": t1 - t0,
        }

    except KeyboardInterrupt as e:
        # Treat KeyboardInterrupt inside worker as an error; the parent
        # will see "error" status and can fall back to SAT.
        data = {
            "error": f"KeyboardInterrupt in worker: {repr(e)}",
        }

    except Exception as e:
        data = {
            "error": repr(e),
        }

    try:
        with open(result_path, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        # If we cannot even write the result, there's nothing more to do.
        pass





def run_rc2_with_timeout_file(
    wcnf: WCNF,
    timeout_s: float,
    debug_prefix: str = "[WISE]",
):
    """
    Run RC2 on 'wcnf' in a separate process with a wall-clock timeout.

    Returns: (model, cost, status)
      - status == "ok"        : model, cost are valid.
      - status == "timeout"   : model, cost are None (worker killed by timeout).
      - status == "error"     : model, cost are None (worker crashed).
      - status == "user_abort": model, cost are None (parent got KeyboardInterrupt
                                while waiting; worker terminated & cleaned up).

    KeyboardInterrupt is *never* propagated outside this function.
    """

    # If timeout <= 0, treat as "no RC2".
    if timeout_s is not None and timeout_s <= 0:
        if debug_prefix:
            print(
                f"{debug_prefix} RC2 disabled (timeout <= 0); treating as timeout.",
                flush=True,
            )
        return None, None, "timeout"

    with tempfile.TemporaryDirectory() as tmpdir:
        wcnf_path = os.path.join(tmpdir, "instance.wcnf.pkl")
        result_path = os.path.join(tmpdir, "result_rc2.pkl")

        # Dump WCNF to file for worker.
        with open(wcnf_path, "wb") as f:
            pickle.dump(wcnf, f)

        p = mp.Process(target=_rc2_worker_from_file,
                       args=(wcnf_path, result_path))
        # Optional: make sure child dies if parent dies badly.
        p.daemon = True

        if debug_prefix:
            print(
                f"{debug_prefix} RC2 worker starting (timeout={timeout_s:.1f}s) "
                f"for WCNF: vars={wcnf.nv}, hard={len(wcnf.hard)}, soft={len(wcnf.soft)}",
                flush=True,
            )

        try:
            p.start()
        except KeyboardInterrupt:
            if debug_prefix:
                print(
                    f"{debug_prefix} RC2 start interrupted by user; "
                    "terminating worker and returning user_abort.",
                    flush=True,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return None, None, "user_abort"

        # Our own timeout loop instead of a single long join(timeout_s)
        start = time.time()
        status = None

        try:
            while True:
                # Short join slice (e.g. 0.5s) so we can
                #  - check elapsed time
                #  - respond quickly to KeyboardInterrupt
                p.join(0.5)

                if not p.is_alive():
                    # Worker finished.
                    status = "finished"
                    break

                elapsed = time.time() - start
                if elapsed >= timeout_s:
                    status = "timeout"
                    break

        except KeyboardInterrupt:
            # Parent got Ctrl-C while waiting.
            if debug_prefix:
                elapsed = time.time() - start
                print(
                    f"{debug_prefix} RC2 join interrupted by user after "
                    f"{elapsed:.3f}s; terminating worker (user_abort).",
                    flush=True,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return None, None, "user_abort"

        if status == "timeout":
            if debug_prefix:
                elapsed = time.time() - start
                print(
                    f"{debug_prefix} RC2 worker exceeded {timeout_s:.1f}s "
                    f"(elapsed={elapsed:.3f}s); terminating (timeout).",
                    flush=True,
                )
            if p.is_alive():
                p.terminate()
                try:
                    p.join(5.0)
                except Exception:
                    pass
            return None, None, "timeout"

        # status == "finished": worker exited within timeout
        if not os.path.exists(result_path):
            if debug_prefix:
                print(
                    f"{debug_prefix} RC2 worker finished but wrote no result file; "
                    "treating as error.",
                    flush=True,
                )
            return None, None, "error"

        try:
            with open(result_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            if debug_prefix:
                print(
                    f"{debug_prefix} RC2 worker result read error: {e!r}; "
                    "treating as error.",
                    flush=True,
                )
            return None, None, "error"

        if "error" in data:
            if debug_prefix:
                print(
                    f"{debug_prefix} RC2 worker raised: {data['error']}; "
                    "treating as error.",
                    flush=True,
                )
            return None, None, "error"

        if debug_prefix:
            print(
                f"{debug_prefix} RC2 worker finished in {data.get('time', 0.0):.3f}s, "
                f"opt_cost={data.get('cost')}",
                flush=True,
            )

        return data.get("model"), data.get("cost"), "ok"


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
    def _optimal_QMR_for_WISE(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,    # unused in D-min version
        wV: List[int] = None,    # unused in D-min version
        wB_col: int = 1,
        wB_row: int = 1,
        max_rc2_time: float = 600.0,
        max_sat_time = 360.0,  
        active_ions: Set[int] = None

    ) -> List[np.ndarray]:
        """
        Three-level optimizer for WISE where this code deals with level 2 onwards:

        Inputs:
        - Initial layout A_in : n×m array of ion indices.
        - Circuit P_arr: list of size R, where for each round r, P_arr[r] is the list of interacting ion pairs in that round.
        - Block size k.
        - Boundary targets BT: list of size R; for each round r and ion i, 
            BT[r][i] = (d, c) means “ion i must end round r in row d and column c”
            (these come from the previous slice’s SAT run and are treated as hard pins here).

        Derived sets (per round r):
        - Res_r  = { i | i ∈ BT[r] }      reserved ions in round r (pinned by BT).
        - Free_r = all ions \ Res_r       non-reserved ions in round r (to be routed in this slice).


        Let:
        - n, m be the slice / subgrid shape (rows × columns),
        - R be the number of lookahead rounds (typically small, e.g. R=2),
        ----------------------------------------------------------------------
        Three-level optimization scheme
  
        Level 1: Spatial slicing and incremental subgrid growth

        - Given a global ion layout for WISE and a sequence of two-qubit rounds, we:
        1. Start with a small subgrid anchored at (0,0) labelled A_in and no ions pinned in BT
        2. Partition the circuit's two-qubit gates into parallel rounds and look only at the next R rounds, where each round has at most (#traps) gates
        2. Iteratively do the following:
            a. Determine P_arr from two-qubit gates that live fully inside the grid across the partitioned circuit.
            a. Run levels 2-3 to place ions within the subgrid.
            b. Pin ions at each round that are involved in a two-qubit gate in the subgrid
            c. Expand the subgrid by increasing its size to the right or downwards.
            d. Remove gates from this iteration from the partitioned circuit

        Level 2: Per-subgrid D-minimising SAT solver

        (Obtain a per-subgrid routing that minimises the maximum displacement D across all rounds and all ions)
        1. Define the search range:
        - D_lo = 0
        - D_hi = max(n−1, m−1) 

        2. For a candidate D_mid in [D_lo, D_hi], build the CNF with:
        - constraints (0)–(7) below,
        - the movement bound set to D_bound = D_mid.

        3. Call a SAT solver on this CNF:
        - If SAT, record D_mid and its model as a candidate solution,
            and tighten the upper bound: D_hi = D_mid − 1.
        - If UNSAT, increase the lower bound: D_lo = D_mid + 1.

        4. At the end of binary search, let D* be the smallest D_mid for which the CNF was SAT.
        If none exists in [0, max(n−1, m−1)], declare UNSAT for the slice.

        Level 3: Boundary-aware MaxSAT at D*

        1. Rebuild the formula at D_bound = D* as a weighted CNF, using the same hard constraints.
        2. Add the boundary-avoidance clauses (8) as soft clauses, with weights w_B_col, w_B_row.
        3. Run RC2 on this WCNF to minimize the number (and weighted importance) of soft clause violations.

        Because D* is fixed, this second level only nudges the solution among all layouts that achieve D*,
        preferring those that keep interacting ions away from the outer boundary.
        ----------------------------------------------------------------------
        Variables (for this slice):
        - a[r, k, j, i]  ion i occupies cell (row k, col j) in layout at round r
                        where r ∈ {0,…,R} (a[0] is A_in, a[1]…a[R] are layouts after each round)
        - x[r, i, c]     in round r, ion i has target column c
        - t[r, i, d]     in round r, ion i has target row d
        - w[r, i, b]     in round r, ion i is in horizontal block b = 0,…,⌈m/k⌉−1
        - p[r, k, i]     in layout a[r], ion i is present in row k
        - y[r, k, c, i]  in round r, ion i is in row k and targets column c

        Hard constraints:

        For a given distance bound D ≥ 0, we build a structural CNF encoding the following:

        (0) Exactly one ion per cell in each layout:
        - For all r ∈ {0,…,R}, for all k ∈ {0,…,n−1}, for all j ∈ {0,…,m−1}:
            ∑_i a[r, k, j, i] = 1

        (1) Initial layout:
        - For all k, j:
            a[0, k, j, A_in[k, j]]

        (2) BT pins and x/t one-hots:
        - For each round r and ion i:

        • If i ∈ Res_r with BT[r][i] = (d_fix, c_fix):
            - x[r, i, c_fix]
            - t[r, i, d_fix]
            - w[r, i, floor(c_fix / k)]
            - a[r+1, d_fix, c_fix, i]
            (the layout after round r has ion i fixed at its pinned cell)

        • If i ∉ Res_r (free ion in this slice):
            - ∑_c x[r, i, c] = 1     (ion chooses exactly one destination column)
            - ∑_d t[r, i, d] = 1     (ion chooses exactly one destination row)

        (3) Block membership (for free ions):
        - For all r, all i ∉ Res_r, all blocks b:
            let block b be columns c ∈ {b·k,…,min((b+1)·k−1, m−1)}.
            w[r, i, b] ↔ (∨_{c in block b} x[r, i, c])

        (4) Pair constraints (same destination row and same block):
        - For all r, for all (i1, i2) ∈ P_arr[r]:
            for all destination rows d:
                t[r, i1, d] ↔ t[r, i2, d]
            for all blocks b:
                w[r, i1, b] ↔ w[r, i2, b]
            (This ensures ions in a 2-qubit gate are in the same trap)

        (5) Rounds glued together (layout consistency across rounds):
        - For all r ∈ {0,…,R−1}, for all ions i, all rows d, all columns c:
            a[r+1, d, c, i] ↔ ( x[r, i, c] ∧ t[r, i, d] )
            (This ensures that the layout after round r is exactly the permutation given by x[r,·,·] and t[r,·,·].)

        (6) Row presence and y linkage (reserved vs free semantics):

        (6a) Row presence p:
        - For all r, k, i:
            p[r, k, i] ↔ (∨_j a[r, k, j, i])
            (ion i is “present in row k” at round r iff it actually occupies some cell in that row)

        (6b) One-per (r, k, c) and y‐link:

        - For all r, k, c:
            ∑_i y[r, k, c, i] ≤ 1
            (at most one ion from row k targets column c in round r)

        - For all r, k, c, i:
            if i ∉ Res_r (free ion):
                y[r, k, c, i] ↔ (x[r, i, c] ∧ p[r, k, i])
            if i ∈ Res_r with BT[r][i] = (d_fix, c_fix):
                - if c = c_fix:
                    y[r, k, c_fix, i] ↔ p[r, k, i]
                - if c ≠ c_fix:
                    y[r, k, c, i] = False
            (helper variable y is linked to variables x and p consistently)

        (Consequences of (6): no two ions from the same current row can target the same column, which makes reconfiguration by WISE control hardware easy.)

        (7) Movement bound with respect to D:
        - For each round r, for each ion i, for each source cell (k, j) where a[r, k, j, i] holds:

        Horizontal bound:
        - For all columns c with |c − j| > D:
            a[r, k, j, i] → ¬x[r, i, c]
            (ion i cannot choose a target column c more than D steps away from its current column j.)

        Vertical bound:
        - For all rows d with |d − k| > D:
            a[r, k, j, i] → ¬t[r, i, d]
            (ion i cannot choose a target row d more than D steps away from its current row k.)

        This enforces:
            max_{r,i} max( |c_dest − c_src|, |d_dest − d_src| ) ≤ D

        and the rest of the constraints guarantee that such a move pattern is globally consistent
        with BT pins and pair constraints.

        (8) Boundary-avoidance (only in the second level, as soft constraints):
        - For all rounds r, for all pairs (i1, i2) ∈ P_arr[r]:
            Soft clauses:
            - ¬x[r, i1, m−1]   with weight w_B_col
            - ¬t[r, i1, n−1]   with weight w_B_row
            - ¬x[r, i2, m−1]   with weight w_B_col
            - ¬t[r, i2, n−1]   with weight w_B_row

        These softly discourage interacting ions from being routed to the last column and last row,
        subject to not violating the hard constraints and the distance bound D. 

        Complexity:
        The asymptotic worst-case looks like O(R n² m²), which is inherent to modeling all ions
        and all cells explicitly, but subgrid slicing means there are many unit clauses (you never feed the full chip to one SAT instance).
        The hard part (finding a feasible layout respecting BT and pairs) is done by SAT, which scales
        more predictably than a heavy weighted MaxSAT.
        The MAXSAT stage operates on the same structural CNF but adds only a small number of soft
        boundary clauses, so its overhead remains close to a single SAT run.
        """

        DEBUG_DIAG = True
        DEBUG_DIAG_DETAILED = False


        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)

        if BT is None:
            BT = [{} for _ in range(R)]

        row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}
        ions_all = set(int(x) for x in A_in.flatten())

        if active_ions is None:
            active_ions=set()
         
            # -------------------------------
            # Identify "active" vs "spectator" ions
            # -------------------------------
            active_ions = set()
            # Any ion in a pair is active
            for r, pairs in enumerate(P_arr):
                for i1, i2 in pairs:
                    if i1 in ions_all:
                        active_ions.add(i1)
                    if i2 in ions_all:
                        active_ions.add(i2)
            # Any ion pinned in BT is also active
            for r, bt in enumerate(BT):
                for i in bt.keys():
                    if i in ions_all:
                        active_ions.add(i)

        spectator_ions = ions_all - set(active_ions)

        if DEBUG_DIAG:
            print(
                f"[WISE] ions_all={len(ions_all)}, "
                f"active_ions={len(active_ions)}, "
                f"spectator_ions={len(spectator_ions)}",
                flush=True,
            )


        # -------------------------------
        # Pre-checks (same semantics as before)
        # -------------------------------

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
        # Structural CNF / WCNF builder
        # -------------------------------
        def build_structural_cnf(
            D_bound: int,
            use_wcnf: bool = False,
            add_boundary_soft: bool = False,
            phase_label: str = "",
        ):
            """
            Build CNF (or WCNF) encoding:

            Vars:
                a[r,k,j,i] : layout at round r (r=0..R)
                x[r,i,c]   : dest column in round r
                t[r,i,d]   : dest row    in round r
                w[r,i,b]   : dest block  in round r
                p[r,k,i]   : ion i present in row k at layout a[r]
                y[r,k,c,i] : ion i in row k targeting col c at round r
                h[r,δ], v[r,δ] : band variables (MaxSAT stage only)

            Hard:
                - a: exactly one ion per cell for r=0..R,
                - initial a[0] equals A_in,
                - x/t one-hot **only for non-reserved ions**,
                - w link only for non-reserved ions,
                - BT pins: x/t/w units + a[r+1,d_fix,c_fix,i],
                - pair equalities on t and w,
                - a[r+1] <-> (x[r], t[r]),
                - p & y semantics (reserved vs free),
                - bound: |c - j| <= D_bound and |d - k| <= D_bound.

            Soft (when use_wcnf=True):
                - band variables penalising larger displacements,
                - optional boundary avoidance on last row/col for interacting ions.
            """

            vpool = IDPool()

            def var_a(r, k, j, i):  return vpool.id(('a', r, k, j, i))
            def var_x(r, i, c):     return vpool.id(('x', r, i, c))
            def var_t(r, i, d):     return vpool.id(('t', r, i, d))
            def var_w(r, i, b):     return vpool.id(('w', r, i, b))
            def var_p(r, k, i):     return vpool.id(('p', r, k, i))
            def var_y(r, k, c, i):  return vpool.id(('y', r, k, c, i))
            def var_h(r, delta):    return vpool.id(('h', r, delta))
            def var_v(r, delta):    return vpool.id(('v', r, delta))

            def is_reserved(r, i):  return i in BT[r]

            stats = Counter()

            if use_wcnf:
                f = WCNF()

                def add_hard(cl, tag=None):
                    if tag is not None:
                        stats[tag] += 1
                    f.append(cl)

                def add_soft(cl, w, tag=None):
                    if tag is not None:
                        stats[tag] += 1
                    f.append(cl, weight=w)
            else:
                f = CNF()

                def add_hard(cl, tag=None):
                    if tag is not None:
                        stats[tag] += 1
                    f.append(cl)

                def add_soft(cl, w, tag=None):
                    raise RuntimeError("soft clauses not allowed in pure CNF")

            # --- high-level debug about this call ---
            if DEBUG_DIAG:
                print(
                    f"[WISE] build_structural_cnf({phase_label}): "
                    f"D_bound={D_bound}, use_wcnf={use_wcnf}, "
                    f"boundary_soft={add_boundary_soft}, n={n}, m={m}, R={R}",
                    flush=True,
                )
                for r in range(R):
                    res_r = sum(1 for i in ions if i in BT[r])
                    free_r = len(ions) - res_r
                    print(
                        f"        round {r}: |Res_r|={res_r}, |Free_r|={free_r}, "
                        f"|P[r]|={len(P_arr[r])}, |BT[r]|={len(BT[r])}",
                        flush=True,
                    )

            # (0) a-cardinality: exactly one ion per cell for r=0..R
            for r in range(R + 1):
                for krow in range(n):
                    for jcol in range(m):
                        lits = [var_a(r, krow, jcol, i) for i in ions]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_hard(cl, tag="a_card")

            # (1) initial layout a[0]
            for krow in range(n):
                for jcol in range(m):
                    ion0 = int(A_in[krow, jcol])
                    add_hard([var_a(0, krow, jcol, ion0)], tag="init")

            # (2) BT units and x/t/w-onehot for **free** ions
            for r in range(R):
                for i in ions:
                    if is_reserved(r, i):
                        # reserved: BT gives units, no equals on x/t/w
                        d_fix, c_fix = BT[r][i]
                        add_hard([var_x(r, i, c_fix)], tag="BT_x")
                        add_hard([var_t(r, i, d_fix)], tag="BT_t")
                        add_hard([var_w(r, i, c_fix // k)], tag="BT_w")
                        # fix layout after this round
                        add_hard([var_a(r + 1, d_fix, c_fix, i)], tag="BT_a_next")
                    else:
                        # free ion: x one-hot
                        lits_x = [var_x(r, i, c) for c in range(m)]
                        encx = CardEnc.equals(lits=lits_x, encoding=EncType.ladder, vpool=vpool)
                        for cl in encx.clauses:
                            add_hard(cl, tag="x_onehot")
                        # free ion: t one-hot
                        lits_t = [var_t(r, i, d) for d in range(n)]
                        enct = CardEnc.equals(lits=lits_t, encoding=EncType.ladder, vpool=vpool)
                        for cl in enct.clauses:
                            add_hard(cl, tag="t_onehot")

            # (3) w-link for free ions (reserved ions got w fixed above)
            for r in range(R):
                for i in ions:
                    if not is_reserved(r, i):
                        for b in range(num_blocks):
                            cols = list(range(b * k, min((b + 1) * k, m)))
                            wv = var_w(r, i, b)
                            add_hard([-wv] + [var_x(r, i, c) for c in cols], tag="w_link")
                            for c in cols:
                                add_hard([-var_x(r, i, c), wv], tag="w_link")

            # (4) pair constraints: same dest row & block
            for r in range(R):
                for (i1, i2) in P_arr[r]:
                    if i1 not in ions or i2 not in ions:
                        continue
                    for d in range(n):
                        add_hard([-var_t(r, i1, d), var_t(r, i2, d)], tag="pair_t")
                        add_hard([-var_t(r, i2, d), var_t(r, i1, d)], tag="pair_t")
                    for b in range(num_blocks):
                        add_hard([-var_w(r, i1, b), var_w(r, i2, b)], tag="pair_w")
                        add_hard([-var_w(r, i2, b), var_w(r, i1, b)], tag="pair_w")

            # (5) glue: a[r+1,d,c,i] <-> (x[r,i,c] & t[r,i,d]) for *all* ions
            for r in range(R):
                for i in ions:
                    for d in range(n):
                        for c in range(m):
                            a_next = var_a(r + 1, d, c, i)
                            xv = var_x(r, i, c)
                            tv = var_t(r, i, d)
                            add_hard([-a_next, xv], tag="glue")
                            add_hard([-a_next, tv], tag="glue")
                            add_hard([-xv, -tv, a_next], tag="glue")

            # (6) p row presence & y linkage
            for r in range(R):
                # (6a) p[r,k,i] <-> OR_j a[r,k,j,i]
                for krow in range(n):
                    for i in ions:
                        A_lits = [var_a(r, krow, j, i) for j in range(m)]
                        add_hard([-var_p(r, krow, i)] + A_lits, tag="p_row")
                        for aj in A_lits:
                            add_hard([-aj, var_p(r, krow, i)], tag="p_row")

                # (6b) y at-most-1 and y-link
                for krow in range(n):
                    for c in range(m):
                        # y at-most-1 per (r,krow,c)
                        y_lits = [var_y(r, krow, c, i) for i in ions]
                        enc = CardEnc.atmost(lits=y_lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_hard(cl, tag="y_atmost")

                        # y linkage, split free vs reserved
                        for i in ions:
                            yv = var_y(r, krow, c, i)
                            pv = var_p(r, krow, i)
                            if not is_reserved(r, i):
                                xv = var_x(r, i, c)
                                add_hard([-yv, xv], tag="y_link_free")
                                add_hard([-yv, pv], tag="y_link_free")
                                add_hard([-xv, -pv, yv], tag="y_link_free")
                            else:
                                d_fix, c_fix = BT[r][i]
                                if c == c_fix:
                                    add_hard([-yv, pv], tag="y_link_res")
                                    add_hard([-pv, yv], tag="y_link_res")
                                else:
                                    add_hard([-yv], tag="y_link_res_zero")

            # (7) movement bound: |c - j| <= D_bound, |d - k| <= D_bound
            if D_bound is not None:
                for r in range(R):
                    for krow in range(n):
                        for jcol in range(m):
                            for i in active_ions:
                                a_src = var_a(r, krow, jcol, i)
                                # horizontal distance
                                for c in range(m):
                                    if abs(c - jcol) > D_bound:
                                        add_hard([-a_src, -var_x(r, i, c)], tag="move_h")
                                # vertical distance
                                for d in range(n):
                                    if abs(d - krow) > D_bound:
                                        add_hard([-a_src, -var_t(r, i, d)], tag="move_v")

            # (8) band variables (Level 2 only): h[r,δ], v[r,δ] and softs
            if use_wcnf and (D_bound is not None) and D_bound > 0:
                BAND_DELTA_CAP = min(D_bound, 3)  # keep small to avoid blow-up

                # 8a) band monotonicity: h[r,δ+1] -> h[r,δ], same for v
                for r in range(R):
                    for delta in range(1, BAND_DELTA_CAP):
                        add_hard([-var_h(r, delta + 1), var_h(r, delta)])
                        add_hard([-var_v(r, delta + 1), var_v(r, delta)])

                # 8b) triggers:
                # If |c-j| = δ and a[r,k,j,i] & x[r,i,c] then h[r,δ],
                # If |d-k| = δ and a[r,k,j,i] & t[r,i,d] then v[r,δ].
                for r in range(R):
                    for krow in range(n):
                        for jcol in range(m):
                            for i in active_ions:
                                a_src = var_a(r, krow, jcol, i)
                                # horizontal movement
                                for c in range(m):
                                    dist = abs(c - jcol)
                                    if 1 <= dist <= BAND_DELTA_CAP:
                                        add_hard([-a_src, -var_x(r, i, c), var_h(r, dist)])
                                # vertical movement
                                for d in range(n):
                                    dist = abs(d - krow)
                                    if 1 <= dist <= BAND_DELTA_CAP:
                                        add_hard([-a_src, -var_t(r, i, d), var_v(r, dist)])

                # 8c) soft clauses: ¬h[r,δ], ¬v[r,δ] with increasing weight in δ
                if wH is not None and len(wH) > BAND_DELTA_CAP:
                    h_weights = wH
                else:
                    h_weights = [0] + [delta for delta in range(1, BAND_DELTA_CAP + 1)]

                if wV is not None and len(wV) > BAND_DELTA_CAP:
                    v_weights = wV
                else:
                    v_weights = [0] + [delta for delta in range(1, BAND_DELTA_CAP + 1)]

                for r in range(R):
                    for delta in range(1, BAND_DELTA_CAP + 1):
                        add_soft([-var_h(r, delta)], h_weights[delta])
                        add_soft([-var_v(r, delta)], v_weights[delta])

            # # (9) optional soft boundary-avoidance
            if use_wcnf and add_boundary_soft and wB_col>0 and wB_row>0:
                for r in range(R):
                    for (i1, i2) in P_arr[r]:
                        if i1 in ions:
                            add_soft([-var_x(r, i1, m - 1)], wB_col, tag="soft_boundary")
                            add_soft([-var_t(r, i1, n - 1)], wB_row, tag="soft_boundary")
                        if i2 in ions:
                            add_soft([-var_x(r, i2, m - 1)], wB_col, tag="soft_boundary")
                            add_soft([-var_t(r, i2, n - 1)], wB_row, tag="soft_boundary")

            # --- final debug dump for this build ---
            if DEBUG_DIAG_DETAILED:
                if use_wcnf:
                    total_hard = len(f.hard)
                    total_soft = len(f.soft)
                else:
                    total_hard = len(f.clauses)
                    total_soft = 0

                print(
                    f"[WISE] build_structural_cnf({phase_label}) stats:"
                    f" vars={vpool.top}, hard={total_hard}, soft={total_soft}",
                    flush=True,
                )
                for tag, count in sorted(stats.items()):
                    print(f"        {tag:16s}: {count} clauses", flush=True)

            return f, vpool, ions, var_a, var_x, var_t
        

        


       

        # -------------------------------
        # Level 1: SAT + binary search on D (global max displacement)
        # -------------------------------
        D_lo = 0
        D_hi = max(n - 1, m - 1)
        best_D = None

        if DEBUG_DIAG:
            print(f"[WISE] starting binary search for D in [0, {D_hi}]", flush=True)

        while D_lo <= D_hi:
            D_mid = (D_lo + D_hi) // 2
            cnf_mid, vpool_mid, ions_mid, var_a_mid, var_x_mid, var_t_mid = \
                build_structural_cnf(
                    D_mid,
                    use_wcnf=False,
                    add_boundary_soft=False,
                    phase_label=f"D={D_mid}/SAT",
                )

            t_sat_start = time.time()
            sat_ok, model_mid, status_sat = run_sat_with_timeout_file(
                cnf_mid,
                timeout_s=max_sat_time,
                debug_prefix="[WISE]",
            )
            t_sat_end = time.time()

            if DEBUG_DIAG:
                print(
                    f"[WISE]  test D={D_mid}: status={status_sat}, SAT={sat_ok}, "
                    f"vars={vpool_mid.top}, clauses={len(cnf_mid.clauses)}, "
                    f"time={t_sat_end - t_sat_start:.3f}s",
                    flush=True,
                )

            if status_sat != "ok":
                # timeout or error
                if best_D is not None:
                    # We already have a feasible D; stop search and use it.
                    if DEBUG_DIAG:
                        print(
                            f"[WISE] SAT at D={D_mid} ended with status={status_sat}; "
                            f"falling back to previous best_D={best_D[0]} and "
                            "stopping binary search.",
                            flush=True,
                        )
                    break
                # else:
                #     # No feasible D found yet => we don't know if instance is solvable
                #     raise RuntimeError(
                #         f"SAT solver failed with status={status_sat} "
                #         "before any feasible D was found."
                #     )

            # Normal SAT result
            if sat_ok:
                # Record this as the best so far and tighten upper bound
                best_D = (D_mid, model_mid, vpool_mid, ions_mid, var_a_mid)
                D_hi = D_mid - 1
            else:
                # UNSAT: need larger D
                D_lo = D_mid + 1

        if best_D is None:
            raise RuntimeError("No feasible layout for any D in [0, max(n-1,m-1)].")


        D_star, sat_model_star, vpool_sat, ions_sat, var_a_sat = best_D
        if DEBUG_DIAG:
            print(f"[WISE] minimal D* found: {D_star}", flush=True)

        # Build WCNF at D* (with bands + boundary softs)
        if DEBUG_DIAG:
            print(f"[WISE] building WCNF at D*={D_star} ...", flush=True)

        t_build_start = time.time()
        wcnf, vpool_w, ions_w, var_a_w, var_x_w, var_t_w = build_structural_cnf(
            D_star,
            use_wcnf=True,
            add_boundary_soft=True,
            phase_label=f"D*={D_star}/WCNF",
        )
        t_build_end = time.time()

        if DEBUG_DIAG:
            print(
                f"[WISE] WCNF built: vars={wcnf.nv}, hard={len(wcnf.hard)}, "
                f"soft={len(wcnf.soft)}, time={t_build_end - t_build_start:.3f}s",
                flush=True,
            )

        model_rc2, cost_rc2, status_rc2 = run_rc2_with_timeout_file(
            wcnf,
            timeout_s=max_rc2_time,
            debug_prefix="[WISE]",
        )

        if DEBUG_DIAG:
            print(
                f"[WISE] RC2 status={status_rc2}, opt_cost={cost_rc2}",
                flush=True,
            )

        # Decide which model to use
        if status_rc2 == "ok" and model_rc2 is not None:
            # Use MaxSAT-refined model
            model_used = model_rc2
            vpool_used = vpool_w
            var_a_used = var_a_w
            ions_used  = ions_w
            if DEBUG_DIAG:
                print("[WISE] using RC2 MaxSAT model at D*", flush=True)
        else:
            # On timeout or error, fall back to SAT model from Level-1
            if DEBUG_DIAG:
                print(
                    "[WISE] MaxSAT unavailable (timeout/error); "
                    "falling back to SAT model at D*.",
                    flush=True,
                )
            model_used = sat_model_star
            vpool_used = vpool_sat
            var_a_used = var_a_sat
            ions_used  = ions_sat

        # Decode layouts a[1]..a[R] from model_used / var_a_used / ions_used
        model_set = {l for l in model_used if l > 0}

        def lit_true(v: int) -> bool:
            return v in model_set

        layouts: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            rr = r + 1  # layout after round r
            for d in range(n):
                for c in range(m):
                    found = None
                    for i in ions_used:
                        if lit_true(var_a_used(rr, d, c, i)):
                            found = i
                            break
                    if found is None:
                        raise RuntimeError(
                            f"could not reconstruct cell (round={rr}, d={d}, c={c})"
                        )
                    nxt[d, c] = found
            layouts.append(nxt)
            cur = nxt

        return layouts




        
    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: QCCDWiseArch,
        arrangement: Mapping[Trap, Sequence[Ion]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        ignoreSpectators: bool = False
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

        spectatorIons = []
        for ions in arrangement.values():
            spectatorIons.extend([ion for ion in ions if isinstance(ion, SpectatorIon)])
  

        # ---------- helper: odd-even passes ----------
        def row_pass_by_rank(even_phase: bool, row_rank: List[Dict[int,int]]) -> bool:
            maxSwapsInRow=0
            start = 0 if even_phase else 1
            for r in range(n):
                swapsInRow=0
                rank = row_rank[r]
                for c in range(start, m-1, 2):
                    a = int(A[r, c]); b = int(A[r, c+1])
                    if not ignoreSpectators and a in spectatorIons and b in spectatorIons:
                        continue
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
                    if not ignoreSpectators and a in spectatorIons and b in spectatorIons:
                        continue
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
                oddpass=col_bucket_pass(True,  t, ion_to_dest_row)
                evenpass=col_bucket_pass(False, t, ion_to_dest_row)
                timeElapsed+=oddpass*col_swap_time
                timeElapsed+=evenpass*col_swap_time
                acc_cost+=int(oddpass)+ int(evenpass)
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
            oddpass = row_pass_by_rank(True,  row_rank_final)
            evenpass = row_pass_by_rank(False, row_rank_final)
            timeElapsed+=oddpass*row_swap_time
            timeElapsed+=evenpass*row_swap_time
            acc_cost+=int(oddpass)+ int(evenpass)

        print(f"RECONFIGURATION: {acc_cost} passes were needed for the current reconfiguration round, taking {timeElapsed} time and {heatingRates} heating")
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
    
