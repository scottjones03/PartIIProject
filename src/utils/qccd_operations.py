import numpy as np
from typing import (
    Sequence,
    List,
    Optional,
    Callable,
    Any,
    Mapping,
    Set,
    Dict,
    Iterable
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
from scipy import stats
from pysat.solvers import Solver
import math

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
        cls, arrangement: Mapping[Trap, Sequence[Ion]], wiseArch: QCCDWiseArch, oldAssignment: Sequence[Sequence[int]], newAssignment: Sequence[Sequence[int]], schedule: List[Dict[str, Any]]
    ):
        heatingRates, reconfigTime = cls._runOddEvenReconfig(wiseArch, arrangement, oldAssignment, newAssignment, sat_schedule=schedule)
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


    @staticmethod
    def _optimal_QMR_for_WISE2(
        A_in: np.ndarray,
        P_arr: List[List[Tuple[int, int]]],
        *,
        k: int,
        BT: List[Dict[int, Tuple[int, int]]] = None,
        wH: List[int] = None,    # unused in D-min version
        wV: List[int] = None,    # unused in D-min version
        wB_col: int = 1,
        wB_row: int = 1,
        max_rc2_time: float = 360.0,
        max_sat_time = 1200.0,  
        active_ions: Set[int] = None,
        freeze_seed_prev: Optional[Dict[str, Any]] = None,
        full_P_arr: List[List[Tuple[int, int]]]=[]
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Three-level optimizer for WISE (Level 2 onwards) with corrected pass-structure
        semantics and clarified BT/pair compatibility constraints.

        Inputs:
        - Initial layout A_in : n×m array of ion indices.
        - Circuit P_arr: list of size R, where for each round r, P_arr[r] is the list
        of interacting ion pairs in that round.
        - Block size k.
        - Boundary targets BT: list of size R; for each round r and ion i,
            BT[r][i] = (d, c) means “ion i must end round r in row d and column c”.
        These come from the previous slice’s SAT run and are treated as hard pins
        in the current slice.

        Derived per round r:
        Res_r  = { i | i ∈ BT[r] }       # reserved ions, fixed at end of r
        Free_r = all ions \ Res_r        # movable ions in this slice

        Parameters:
        n, m = subgrid shape (rows × columns)
        R    = number of lookahead rounds (small, e.g. 2)
        P_max = maximum allowed number of micro-passes per round
                (typically ≈ n + m, architecture-dependent)

        ----------------------------------------------------------------------
        Three-level optimization scheme

        Level 1 — Spatial slicing and incremental subgrid growth:
        (unchanged; omitted for brevity)

        ----------------------------------------------------------------------

        Level 2 — Pass-minimising SAT solver (corrected semantics)
        
        We minimise the number of odd–even micro-passes P needed to route all
        R rounds inside this slice. The key correction is that WISE performs
        **two distinct phases per round**:

            1. pH horizontal odd–even passes (row-wise nearest-neighbour exchanges)
            2. pV vertical odd–even passes   (within-block, column-bucket exchanges)

        These phases must occur strictly in that order; vertical movement is
        impossible during the horizontal phase, and horizontal movement is
        impossible during the vertical phase. Total passes satisfy:

            pH + pV  ≤  P_bound   (P_bound is the guess tested in SAT)

        To determine feasibility, we perform a binary search:

        P_lo = 0
        P_hi = P_max
        while P_lo ≤ P_hi:
            P_mid = floor((P_lo + P_hi)/2)
            Build CNF with P_bound = P_mid
            If SAT:
                record P_mid and tighten: P_hi = P_mid − 1
            else:
                P_lo = P_mid + 1

        The smallest SAT value P* is the slice-optimal pass budget.
        ----------------------------------------------------------------------
        
        Level 2.5 - Adaptive core freezing (this section):
            - Reuse a subset of previous a/s_h/s_v assignments as hard units
            in the new CNF,
            - Maintain BT pins throughout,
            - Start with a large, fully frozen core (MODE_FULL),
            - If UNSAT, gradually unfreeze comparators (MODE_POS_ONLY) and
            shrink the core, ultimately allowing full freedom (MODE_NONE)
            if necessary.

            For each P_bound explored in the binary search (Level 2), we do *not*
            directly call SAT on BaseCNF. Instead, we run an inner adaptive loop:

            Given:
                • P_bound,
                • initial core size (n_core_0, m_core_0),
                • initial freeze horizon P_freeze_0 (≤ P_prev),
                • Freeze modes in order of decreasing rigidity:
                    MODE_FULL  →  MODE_POS_ONLY  →  MODE_NONE.

            We attempt:

                for n_core in [n_core_0, n_core_0 - 1, …, 0]:
                for m_core in [m_core_0, m_core_0 - 1, …, 0]:
                    for mode in [MODE_FULL, MODE_POS_ONLY, MODE_NONE]:

                    build BaseCNF(P_bound, …) with current A_in, P_arr, BT
                    build FreezeUnits using (n_core, m_core, P_freeze_0, mode, vpool_prev, model_prev)

                    let CNF_try = BaseCNF ∧ FreezeUnits
                    if CNF_try is SAT:
                        return SAT, model_try, (n_core, m_core, mode)

                if all attempts are UNSAT:
                    return UNSAT for this P_bound

        ----------------------------------------------------------------------
     
        Level 3: MaxSAT Refinement at Optimal Pass Count

        Once the minimal pass count P* is found (and a satisfying assignment for that case), the algorithm optionally performs a MaxSAT optimization to improve the quality of the solution under the same pass budget. We rebuild the formula for P_bound = P*, but this time as a weighted MaxSAT (WCNF) with additional soft constraints (preferences) that the solver will try to satisfy:
            •	Soft Constraint: Boundary avoidance – We add a penalty if any interacting ion from P_arr[r] ends up on a boundary row or column of the subgrid. Intuitively, we prefer pairs to meet in more central locations (avoiding the last row or last column) if possible, to leave room for future moves. The solver will try to avoid placing gate pairs on the outer boundary by assigning a cost to such outcomes.
            •	Soft Constraint: Swap minimization – We penalize each active swap (each s_h or s_v set to True) to prefer solutions that accomplish the routing with fewer actual exchanges (even if the number of passes is fixed, there might be different swap patterns). This encourages using identity (no-swap) operations where possible, effectively minimizing the total number of SWAP gates executed. Reducing swaps can directly improve fidelity and reduce error in quantum circuits ￼.

        These soft constraints are added to the CNF as clauses that can be violated at a cost, and a MaxSAT solver (RC2 algorithm in PySAT, via run_rc2_with_timeout_file) is used to find an assignment that satisfies all the hard constraints (so still achieves the routing in P* passes) while minimizing the total cost of soft constraint violations. In other words, it finds an optimal trade-off where, for example, it might allow one pair on a boundary if that drastically cuts swaps, or vice versa, but tries to satisfy both if possible.
 

        ----------------------------------------------------------------------

        Variables

            Let:

            • n, m = number of rows and columns in the subgrid,
            • R = number of lookahead rounds,
            • P_bound = SAT candidate for max micro-passes per round (0..P_max),
            • ions = set of all ion indices present in A_in.

            Layout variables (position of ion i):

            a[r, p, k, j, i]  Boolean
                “Ion i occupies row k, column j at micro-pass p of round r.”

            Indices:
                r ∈ {0,…,R}              (round index, with r = R the final post-round state)
                p ∈ {0,…,P_bound}        (micro-pass index within a round; for r = R, only p = 0)
                k ∈ {0,…,n−1}, j ∈ {0,…,m−1}
                i ∈ ions

            Interpretation:

            • For r < R:
                a[r,0,:,:,:]   is the layout at **start** of round r.
                a[r,P_bound,:,:,:] is the layout at **end** of round r.
            • a[0,0,:,:,:] is the given A_in.
            • a[r+1,0,:,:,:] is chained to a[r,P_bound,:,:,:].
            • a[R,0,:,:,:] is the post-round-R final layout.

        Comparator variables:

            Horizontal comparators (within a row):
                s_h[r, p, k, j]  Boolean
                    “During round r, at pass p, apply a horizontal comparator between
                    cells (k,j) and (k,j+1).”

                r ∈ {0,…,R−1}
                p ∈ {0,…,P_bound−1}
                k ∈ {0,…,n−1}
                j ∈ {0,…,m−2}

            Vertical comparators (per column, symmetric to s_h):

                s_v[r, p, k, j]  Boolean
                    “During round r, at pass p, apply a vertical comparator between
                    cells (k, j) and (k+1, j).”

                r ∈ {0,…,R−1}
                p ∈ {0,…,P_bound−1}
                k ∈ {0,…,n−2}
                j ∈ {0,…,m−1}
                
            Phase selection variables:

                phase[p]  Boolean
                    “Pass p is in the vertical phase (phase[p]=1) or horizontal phase
                    (phase[p]=0).”

                Indices:
                    p ∈ {0,…,P_bound−1} (same phases for all rounds; or can be duplicated
                                            per round r if desired, but here we use global p.)

                The encoding makes phase[p] **monotone non-decreasing**, so there exists
                an implicit p_H (number of horizontal passes) such that:

                • for passes p < p_H  : phase[p] = 0  (horizontal),
                • for passes p ≥ p_H  : phase[p] = 1  (vertical),

                and thus p_H + p_V ≤ P_bound.

            End-of-round abstraction variables:

                row_end[r, i, d]  Boolean
                    “At the end of round r, ion i is in row d.”

                w_end[r, i, b]    Boolean
                    “At the end of round r, ion i is in horizontal block b
                    (block b: columns j ∈ [b·k, min((b+1)·k−1, m−1)]).”

                Indices:
                    r ∈ {0,…,R−1}
                    i ∈ ions
                    d ∈ {0,…,n−1}
                    b ∈ {0,…,num_blocks−1}

            BT pin data (given, not variables):

            BT[r][i] = (d_fix, c_fix)  for some rounds r and ions i, meaning:
                “Ion i is required to be at row d_fix, col c_fix at the end of round r
                (i.e., at micro-pass p = P_bound in round r).”

        ----------------------------------------------------------------------  
        Hard Constraints

        For each SAT call at a chosen P_bound, we build the CNF to enforce:

        (0) Global permutation at every time-step.
        (1) Initial layout and inter-round chaining.
        (2) BT pinning at end-of-round (final-state only).
        (3) Horizontal vs vertical phase separation.
        (4) Horizontal odd–even semantics (with explicit copy constraints).
        (5) Vertical odd–even semantics (with explicit copy constraints).
        (6) End-of-round row/block abstraction and pair constraints.

        Below we specify each set with explicit CNF-style clauses.

        ------------------------------------------------------------------  
        (0) Global permutation: exactly one ion per cell, each ion in one cell  
        ------------------------------------------------------------------  

            For every round r and pass p where a layout is defined:

            • r ∈ {0,…,R−1}, p ∈ {0,…,P_bound}
            • plus final state r = R, p = 0

            we require:

            (0a) Each cell (k,j) contains exactly one ion i:

                For each (r,p,k,j):
                    Exactly-one over { a[r,p,k,j,i] | i ∈ ions }.

                Encoded via CardEnc.equals, which provides:
                    – “at least one”:  ∨_i a[r,p,k,j,i]
                    – “at most one”:   ¬a_i ∨ ¬a_j  for all distinct i,j.

            (0b) Each ion i occupies exactly one cell (k,j):

                For each (r,p,i):
                    Exactly-one over { a[r,p,k,j,i] | 0 ≤ k < n, 0 ≤ j < m }.

                Again encoded via CardEnc.equals or a ladder encoding.

            Given |ions| = n·m in the slice, (0a)+(0b) together enforce that at each
            (r,p) the mapping (k,j) ↔ i is a *permutation* with no collisions and no
            missing ions.

        ------------------------------------------------------------------  
        (1) Initial layout and inter-round chaining  
        ------------------------------------------------------------------  

            (1a) Initial layout:

            For all cells (k,j), let A_in[k,j] = ion i_0:

                a[0, 0, k, j, i_0] is True,
                a[0, 0, k, j, i] is False for all i ≠ i_0.

            This can be encoded as unit clauses:

            • ( a[0,0,k,j,A_in[k,j]] )
            • for each i ≠ A_in[k,j], ( ¬a[0,0,k,j,i] )

            (1b) Round chaining:

            For r = 0,…,R−1, the start of round r+1 equals the final layout of r:

                ∀k,j,i:  a[r+1, 0, k, j, i] ↔ a[r, P_bound, k, j, i].

            In CNF:

            • (¬a[r+1,0,k,j,i] ∨  a[r,P_bound,k,j,i])
            • (¬a[r,P_bound,k,j,i] ∨  a[r+1,0,k,j,i])

            Final state:

            • The “post-R” layout is a[R,0,:,:,:], which is constrained only by
                chaining from round R−1 and (optionally) any global constraints
                afterwards.

        ------------------------------------------------------------------  
        (2) BT pinning at end-of-round (FINAL STATE ONLY)  
        ------------------------------------------------------------------  

            For each round r and each ion i reserved in that round (i ∈ Res_r), with
            BT[r][i] = (d_fix, c_fix), we **only** constrain the final state
            p = P_bound:

            (2a) Ion i must occupy (d_fix, c_fix) at end-of-round r:

                a[r, P_bound, d_fix, c_fix, i] is True.

                CNF:  ( a[r, P_bound, d_fix, c_fix, i] )

            (2b) Ion i cannot occupy any other cell at end-of-round r:

                For all (k,j) ≠ (d_fix, c_fix):

                    ¬a[r, P_bound, k, j, i]

                CNF:  ( ¬a[r, P_bound, k, j, i] )

            Crucial **correction** relative to earlier formulations:

            • We do *not* treat pinned ions as immobile mid-round.
            • There are **no extra constraints** that disable s_h or s_v once the
                ion reaches (d_fix,c_fix).
            • The pinned ion can move freely during passes p = 0,…,P_bound−1.
            • Only at p = P_bound do we enforce the BT location.

            Conflict with gates can still arise if, for a pair (i1,i2) ∈ P_arr[r],
            BT pins them to incompatible rows or blocks (see (6)), but this is by
            design and handled as UNSAT at the level of the instance, not via
            over-constrained mid-round semantics.

        ------------------------------------------------------------------  
        (3) Phase structure: horizontal → vertical  
        ------------------------------------------------------------------  

            We use phase[p] to indicate whether pass p is vertical (1) or horizontal (0).
            We enforce:

            (3a) Monotonicity (once we switch to vertical, we never go back):

                For p = 0,…,P_bound−2:
                    phase[p] → phase[p+1]

                CNF: (¬phase[p] ∨ phase[p+1])

            This ensures there exists some p_H (possibly 0 or P_bound) such that:

            • phase[p] = 0 for p < p_H,
            • phase[p] = 1 for p ≥ p_H.

            Thus each round consists of:

            • horizontal phase: passes p < p_H,
            • vertical phase: passes p ≥ p_H.

            (3b) Gating horizontal vs vertical comparators:

            • If phase[p] = 0 (horizontal phase), vertical comparators must be off:
                    phase[p] = 0 ⇒ ¬s_v[r,p,krow,b]

                In CNF: ( phase[p] ∨ ¬s_v[r,p,krow,b] )

                When phase[p] = 0, this clause reduces to (0 ∨ ¬s_v) ⇒ ¬s_v.
                When phase[p] = 1, clause is satisfied and places no restriction.

            • If phase[p] = 1 (vertical phase), horizontal comparators must be off:
                    phase[p] = 1 ⇒ ¬s_h[r,p,k,j]

                In CNF: ( ¬phase[p] ∨ ¬s_h[r,p,k,j] )

                When phase[p] = 1, clause is (0 ∨ ¬s_h) ⇒ ¬s_h.
                When phase[p] = 0, it is satisfied and does nothing.

            Together, (3a)+(3b) enforce a strict “all horizontals first, then all
            verticals” structure, but let SAT decide how many passes belong to each
            phase within the budget P_bound.

        ------------------------------------------------------------------  
        (4) Horizontal odd–even semantics (with copy constraints)  
        ------------------------------------------------------------------  

            We assume that in any pass p where phase[p] = 0, horizontal comparators
            can be active, respecting odd–even parity on columns, and must implement
            either a swap or identity on their endpoints.

            Parity (no conflicting comparators):

            For passes in horizontal phase, we define a parity index h_index(p) as
            the number of horizontal passes up to p (or simply use p itself if we
            fix which passes are horizontal). Conceptually:

                • if h_index(p) is even: comparators starting at even columns j are
                allowed (0,2,4,…); odd j are disabled.
                • if h_index(p) is odd: comparators at odd columns j are allowed;
                even j are disabled.

            Formally (even though h_index is a derived notion), we enforce:

            (4a) For each (r,p,k,j):

                If j is not allowed by parity for this pass, then s_h[r,p,k,j] = 0:

                    ( ¬allowed_h(r,p,k,j) ∨ ¬s_h[r,p,k,j] )

                where allowed_h can be implemented either by:
                    – using p%2 directly as the phase-local parity,
                    – or counting horizontal passes explicitly (more complex).

            Horizontal swap semantics at (r,p,k,j):

                Let:
                    • left cell  = (k,j)
                    • right cell = (k,j+1)
                    • s = s_h[r,p,k,j]

                For each ion i, we encode *forward* semantics only, relying on global
                cardinality constraints to ensure consistency and permutation behaviour.

                (H1) If s=0, left cell contents persist:
                    (¬s ∧ a[r,p,k,j,i]) → a[r,p+1,k,j,i]

                    CNF: ( s ∨ ¬a[r,p,k,j,i] ∨ a[r,p+1,k,j,i] )

                (H2) If s=0, right cell contents persist:
                    (¬s ∧ a[r,p,k,j+1,i]) → a[r,p+1,k,j+1,i]

                    CNF: ( s ∨ ¬a[r,p,k,j+1,i] ∨ a[r,p+1,k,j+1,i] )

                (H3) If s=1, right cell moves to left at next step:
                    (s ∧ a[r,p,k,j+1,i]) → a[r,p+1,k,j,i]

                    CNF: ( ¬s ∨ ¬a[r,p,k,j+1,i] ∨ a[r,p+1,k,j,i] )

                (H4) If s=1, left cell moves to right at next step:
                    (s ∧ a[r,p,k,j,i]) → a[r,p+1,k,j+1,i]

                    CNF: ( ¬s ∨ ¬a[r,p,k,j,i] ∨ a[r,p+1,k,j+1,i] )

            We **do not** encode the reverse implications (from a[r,p+1,…] back to
            a[r,p,…]) because doing so together with global “exactly-one” constraints
            over all ions and all cells can unintentionally forbid any actual swap.
            Instead, we rely on:

            • (0a)/(0b): each cell and each ion is unique at every time-step,
            • (H1)–(H4): if s=0, both endpoints are constrained to preserve their
                contents; if s=1, the contents must flow across endpoints.

            This combination is sufficient to guarantee that:

            • if s=0, the two cells behave as identity on that pass,
            • if s=1, the two cells behave as a simple swap,
            • no ion can disappear or duplicate because of the cardinality clauses.

            Horizontal copy constraints for non-participating cells:

                Consider a fixed round r and pass p with phase[p]=0. For each row k:

                    • If cell (k,j) is not an endpoint of any active comparator at p
                    (i.e., there is no s_h[r,p,k,j'] such that j is j' or j'+1, due to
                    parity and boundary effects), then that cell must copy forward:

                    For all ions i:

                        a[r,p,k,j,i] ↔ a[r,p+1,k,j,i].

                    In CNF:
                        ( ¬a[r,p,k,j,i] ∨  a[r,p+1,k,j,i] )
                        ( ¬a[r,p+1,k,j,i] ∨  a[r,p,k,j,i] )

                Because parity ensures that each cell participates in at most one
                comparator per pass, we can mechanically identify for each (k,j) whether
                it is:

                • left endpoint (k,j) of s_h[r,p,k,j],
                • right endpoint (k,j) of s_h[r,p,k,j−1],
                • or non-participating (in which case the copy constraints apply).

        ------------------------------------------------------------------  
        (5) Vertical odd–even semantics (with copy constraints)  
        ------------------------------------------------------------------  

            Vertical passes now operate symmetrically to horizontal passes, but along
            the **column direction**. At a vertical pass (phase[p] = 1), comparators

                s_v[r, p, k, j]

            swap or preserve the two cells (k, j) and (k+1, j). That is:

                s_v[r,p,k,j] = “During round r, at pass p, apply a vertical comparator
                                between cells (k, j) and (k+1, j).”

            Indices:

                r ∈ {0,…,R−1}
                p ∈ {0,…,P_bound−1}
                k ∈ {0,…,n−2}
                j ∈ {0,…,m−1}

            Parity:

            For each pass p in the vertical phase, let v_index(p) be the count of
            vertical passes up to p (or equivalently p modulo 2 inside the vertical
            phase). Then:

                • If v_index(p) is even: comparators between even row pairs
                    (0–1), (2–3), (4–5), … are allowed (for all j).
                • If v_index(p) is odd: comparators between odd row pairs
                    (1–2), (3–4), (5–6), … are allowed.

            Thus for each (r,p,k,j):

                If k % 2 ≠ v_index(p) % 2, then
                    s_v[r,p,k,j] = 0
                encoded as:
                    ( ¬allowed_v(r,p,k) ∨ ¬s_v[r,p,k,j] ).


            Vertical swap semantics at (r,p,k,j):

            Let:

                top cell    = (k,   j)
                bottom cell = (k+1, j)
                s           = s_v[r,p,k,j].

            For each ion i, the semantics are encoded using one-way (forward)
            implications, identical in structure to the horizontal case:

            (V1) If s = 0, top cell contents persist:
                (¬s ∧ a[r,p,k,j,i]) → a[r,p+1,k,j,i]
                CNF: ( s ∨ ¬a[r,p,k,j,i] ∨ a[r,p+1,k,j,i] )

            (V2) If s = 0, bottom cell contents persist:
                (¬s ∧ a[r,p,k+1,j,i]) → a[r,p+1,k+1,j,i]
                CNF: ( s ∨ ¬a[r,p,k+1,j,i] ∨ a[r,p+1,k+1,j,i] )

            (V3) If s = 1, bottom moves to top:
                (s ∧ a[r,p,k+1,j,i]) → a[r,p+1,k,j,i]
                CNF: ( ¬s ∨ ¬a[r,p,k+1,j,i] ∨ a[r,p+1,k,j,i] )

            (V4) If s = 1, top moves to bottom:
                (s ∧ a[r,p,k,j,i]) → a[r,p+1,k+1,j,i]
                CNF: ( ¬s ∨ ¬a[r,p,k,j,i] ∨ a[r,p+1,k+1,j,i] )

            As in the horizontal case, we do **not** encode reverse implications
            from a[r,p+1,…] to a[r,p,…]. Global cardinality constraints over all ions
            and all cells guarantee that:

                • no ion duplicates or disappears,
                • the only possible behaviours for the pair of cells ((k,j),(k+1,j))
                are identity (s=0) or swap (s=1).


            Vertical copy constraints for non-participating cells:

            For each vertical pass p (phase[p] = 1) and each cell (k,j):

                • If (k,j) is *not* an endpoint of any active vertical comparator at
                that pass (i.e. there is no k' such that (k,j) is (k',j) or (k'+1,j)),
                then the cell must copy its contents forward:

                    a[r,p,k,j,i] ↔ a[r,p+1,k,j,i]

                CNF:
                    ( ¬a[r,p,k,j,i] ∨  a[r,p+1,k,j,i] )
                    ( ¬a[r,p+1,k,j,i] ∨  a[r,p,k,j,i] )

            Parity ensures that each cell participates in at most one vertical
            comparator per pass, so the set of non-participating cells is uniquely
            determined, and their forward propagation is well-defined.

        ------------------------------------------------------------------  
        (6) End-of-round row/block abstraction and pair constraints  
        ------------------------------------------------------------------  

            At the end of each round r (p = P_bound), we must relate the a-variables
            to row_end and w_end, and then enforce that all gate pairs (i1,i2) ∈ P_arr[r]
            end in the same row and block.

            (6a) row_end linkage:

            For each ion i and row d:

                row_end[r, i, d] ↔ (∨_{j=0..m−1} a[r, P_bound, d, j, i])

            CNF:

                (6a-1) row_end[r,i,d] → OR_j a[r,P_bound,d,j,i]:

                ( ¬row_end[r,i,d] ∨ a[r,P_bound,d,0,i] ∨ … ∨ a[r,P_bound,d,m−1,i] )

                (6a-2) a[r,P_bound,d,j,i] → row_end[r,i,d] for each j:

                ( ¬a[r,P_bound,d,j,i] ∨ row_end[r,i,d] )

            (6b) w_end (block) linkage:

            For each ion i and block b, define the set of cells in that block:

                Cells(b) = { (d,j) | 0 ≤ d < n, j_start ≤ j ≤ j_end }

            where j_start = b·k, j_end = min((b+1)·k−1, m−1).

            Then:

            w_end[r, i, b] ↔ (∨_{(d,j) ∈ Cells(b)} a[r,P_bound,d,j,i])

            CNF:

                (6b-1) w_end[r,i,b] → OR_{(d,j) ∈ Cells(b)} a[r,P_bound,d,j,i]:

                ( ¬w_end[r,i,b] ∨ ⋁_{(d,j)∈Cells(b)} a[r,P_bound,d,j,i] )

                (6b-2) For each (d,j) ∈ Cells(b):

                ( ¬a[r,P_bound,d,j,i] ∨ w_end[r,i,b] )

            If Cells(b) is empty (edge case for partially filled last block),
            we simply force w_end[r,i,b] = False.

            (6c) Pair constraints:

            For each gate pair (i1, i2) ∈ P_arr[r]:

                • Same final row:
                    For all d ∈ {0,…,n−1}:

                    row_end[r,i1,d] ↔ row_end[r,i2,d]

                CNF:

                    ( ¬row_end[r,i1,d] ∨ row_end[r,i2,d] )
                    ( ¬row_end[r,i2,d] ∨ row_end[r,i1,d] )

                Because row_end is a one-hot encoding over d (each ion is in exactly
                one row), this equivalence enforces that i1 and i2 end in the *same*
                row.

                • Same final block:
                    For all b ∈ {0,…,num_blocks−1}:

                    w_end[r,i1,b] ↔ w_end[r,i2,b]

                CNF:

                    ( ¬w_end[r,i1,b] ∨ w_end[r,i2,b] )
                    ( ¬w_end[r,i2,b] ∨ w_end[r,i1,b] )

                Since w_end is one-hot over blocks, this forces both ions to end in
                the same horizontal block.

            The combination of (6a)–(6c), plus the permutations enforced by (0)–(5),
            ensures that *for each gate round r* the ions in each two-qubit gate end
            in the same row and the same k-wide block, i.e. on the same WISE trap.

        ------------------------------------------------------------------  
        (7) Frozen-core constraints for subgrid growth  
        ------------------------------------------------------------------  

        When enlarging the subgrid from (n_prev, m_prev) to (n, m), with
        n ≥ n_prev and m ≥ m_prev, we permit reuse of a previous satisfying
        assignment by fixing a top-left “frozen core” region of size
        n_core × m_core, where:

            • 0 ≤ n_core ≤ n_prev
            • 0 ≤ m_core ≤ m_prev

        and fixing micro-passes p = 0,…,P_freeze with
        0 ≤ P_freeze ≤ P_bound.

        Let old_true(X) ∈ {0,1} denote the truth value taken by variable X
        in the previous solving instance.

        A freeze mode MODE ∈ { FULL, POS_ONLY, NONE } determines which
        variables are frozen:

            • MODE = FULL:
                  freeze a-variables, s_h-variables, s_v-variables.

            • MODE = POS_ONLY:
                  freeze a-variables only.

            • MODE = NONE:
                  no freeze constraints are added.

        These clauses apply only to indices inside the frozen region:

            k < n_core,     j < m_core,     p ≤ P_freeze.

        ------------------------------------------------------------------  
        (7a) Frozen position variables a[r,p,k,j,i]  
        ------------------------------------------------------------------  

        For each r ∈ {0,…,R−1}, each p ∈ {0,…,P_freeze},
        each k < n_core, each j < m_core, each ion i:

            If old_true(a[r,p,k,j,i]) = 1, then:

                ( a[r,p,k,j,i] )

            No negative unit clauses for a-variables are added; uniqueness
            is preserved by (0a)/(0b).

        ------------------------------------------------------------------  
        (7b) Frozen horizontal comparators s_h[r,p,k,j]  (MODE = FULL)  
        ------------------------------------------------------------------  

        For each r ∈ {0,…,R−1}, each p ∈ {0,…,P_freeze−1},
        each k < n_core, each j < m_core−1:

            If old_true(s_h[r,p,k,j]) = 1:

                ( s_h[r,p,k,j] )

            else:

                ( ¬s_h[r,p,k,j] )

        ------------------------------------------------------------------  
        (7c) Frozen vertical comparators s_v[r,p,k,j]  (MODE = FULL)  
        ------------------------------------------------------------------  

        For each r ∈ {0,…,R−1}, each p ∈ {0,…,P_freeze−1},
        each k < n_core−1, each j < m_core:

            If old_true(s_v[r,p,k,j]) = 1:

                ( s_v[r,p,k,j] )

            else:

                ( ¬s_v[r,p,k,j] )

        ------------------------------------------------------------------  
        (7d) Interaction with BT pins  
        ------------------------------------------------------------------  

        BT pin constraints in (2) apply unchanged.  
        If a BT pin (i,d_fix,c_fix) lies inside the frozen region and
        contradicts a frozen unit clause, the resulting instance becomes
        UNSAT; resolving this requires reducing n_core or m_core in the
        next attempt.  Pins outside the frozen region are unaffected.

        ------------------------------------------------------------------  
        (7e) Combined formula  
        ------------------------------------------------------------------  

        For a chosen MODE and freeze parameters, the final SAT instance is:

            BaseCNF   ∧   FreezeCNF(n_core, m_core, P_freeze, MODE),

        where BaseCNF denotes the conjunction of constraints (0)–(6).

       ----------------------------------------------------------------------
        Complexity:
            O(R · P_bound · n · m · #ions)

        The combination of:
            • correct horizontal→vertical phase ordering,
            • mandatory vertical connectivity,
            • BT-consistent comparator disabling,
            • strict end-of-round pair co-location
        ensures correctness and eliminates prior UNSAT behaviour caused by
        missing structural constraints.

        The SAT stage finds the minimal number of passes P*. The MaxSAT stage
        optimises within feasible assignments without altering P*.
        """

        DEBUG_DIAG = True
        DEBUG_DIAG_DETAILED = False

        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)

        if BT is None:
            BT = [{} for _ in range(R)]
        if len(full_P_arr)==0:
            full_P_arr=P_arr


        FREEZE_FULL     = "FULL"
        FREEZE_POS_ONLY = "POS_ONLY"
        FREEZE_NONE     = "NONE"

        # -------------------------------
        # Basic sets of ions
        # -------------------------------
        row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}
        ions_all = set(int(x) for x in A_in.flatten())

        if active_ions is None:
            # Identify "active" vs "spectator" ions
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
        # Pre-checks (same semantics as original)
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
        # Structural CNF / WCNF builder for given P_bound
        # -------------------------------
        def _build_structural_cnf(
            P_bound: int,
            use_wcnf: bool = False,
            add_boundary_soft: bool = False,
            phase_label: str = "",
            debug_skip_pair_constraints: bool = False,
            debug_allow_phase_flips: bool = False,
            freeze_params: Optional[Dict[str, Any]]=None
        ):
            """
            Build CNF (or WCNF) encoding for a given pass budget P_bound, capturing rounds 0..R-1 (inclusive).

            Variables:
                a[r, p, k, j, i] : Ion i occupies cell (row=k, col=j) at micro-pass p of round r.
                                r = 0..R, where r=R is final state (after last round)
                                p = 0..P_bound for r < R, and p=0 for r=R.
                s_h[r, p, k, j]  : Horizontal comparator between (k,j) and (k,j+1) at pass p of round r.
                s_v[r, p, k, j]  : Vertical comparator between (k,j) and (k+1,j) at pass p of round r.
                phase[p]         : 0 = horizontal phase, 1 = vertical phase (monotone non-decreasing across p).

            Hard constraints:
                (0) One ion per cell at every micro-pass state and final state.
                (1) Initial layout fixed; end-of-round layout chaining between rounds.
                (2) BT pins: reserved ions fixed to target cell at end of round (p = P_bound).
                (3) Phase structure: phase[p] monotone; phase[p]=0 => horizontal-phase, phase[p]=1 => vertical-phase.
                    Horizontal/vertical comparators masked accordingly.
                (4) Comparator semantics: s_h / s_v implement swap-or-not on their endpoints; this plus (0) makes global permutation.
                (5) End-of-round row/block membership: interacting ions end in same row and same block.
                (6) (Level 3 soft) Boundary avoidance and swap minimisation (if use_wcnf).
                (7) Frozen-core constraints for subgrid growth (optional, via freeze_params).
            """
            vpool = IDPool()

            # ------------- variable helpers -------------

            def var_a(r, p, krow, jcol, ion):
                return vpool.id(("a", r, p, krow, jcol, ion))

            def var_s_h(r, p, krow, jcol):
                return vpool.id(("s_h", r, p, krow, jcol))

            def var_s_v(r, p, krow, jcol):
                return vpool.id(("s_v", r, p, krow, jcol))

            def var_phase(p):
                # phase[p] = 0 => horizontal passes
                # phase[p] = 1 => vertical passes (and once 1, remains 1 unless debug_allow_phase_flips)
                return vpool.id(("phase", p))

            def var_row_end(r, ion, d):
                return vpool.id(("row_end", r, ion, d))

            def var_w_end(r, ion, b):
                return vpool.id(("w_end", r, ion, b))

            def is_reserved(r, ion):
                return ion in BT[r]
            
            def ion_in_full_P_arr(r, ion):
                return any((ion in g) for g in full_P_arr[r])
            
            def ion_in_minor_P_arr(r, ion):
                return any((ion in g) for g in P_arr[r]) or is_reserved(r, ion)

            # ------------- choose CNF or WCNF -------------

            if use_wcnf:
                formula = WCNF()

                def add_clause(cl):
                    formula.append(cl)

                def add_soft_clause(cl, weight=1):
                    formula.append(cl, weight=weight)

            else:
                formula = CNF()

                def add_clause(cl):
                    formula.append(cl)

                def add_soft_clause(cl, weight=1):
                    raise RuntimeError("Soft clauses not allowed in pure CNF mode")

            # if DEBUG_DIAG:
            #     print(
            #         f"[WISE] build_structural_cnf({phase_label}): "
            #         f"P_bound={P_bound}, use_wcnf={use_wcnf}, "
            #         f"boundary_soft={add_boundary_soft}, n={n}, m={m}, R={R}",
            #         flush=True,
            #     )
            #     for r in range(R):
            #         res_r = sum(1 for i in ions if i in BT[r])
            #         free_r = len(ions) - res_r
            #         print(
            #             f"        round {r}: |Res_r|={res_r}, |Free_r|={free_r}, "
            #             f"|P[r]|={len(P_arr[r])}, |BT[r]|={len(BT[r])}",
            #             flush=True,
            #         )

            # ------------------------------------------------------------------
            # (0) Global permutation: exactly one ion per cell AND each ion in exactly one cell
            # ------------------------------------------------------------------

            # (0a) Exactly one ion per cell (r,p,k,j)
            for r in range(R):
                for p in range(P_bound + 1):
                    for krow in range(n):
                        for jcol in range(m):
                            lits = [var_a(r, p, krow, jcol, ion) for ion in ions]
                            enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                            for cl in enc.clauses:
                                add_clause(cl)

            # Final state r = R, p = 0: also exactly-one per cell
            for krow in range(n):
                for jcol in range(m):
                    lits = [var_a(R, 0, krow, jcol, ion) for ion in ions]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_clause(cl)

            # (0b) Each ion occupies exactly one cell (k,j) at every (r,p)
            for r in range(R):
                for p in range(P_bound + 1):
                    for ion in ions:
                        lits = [var_a(r, p, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_clause(cl)

            # final state r=R, p=0 also: each ion in exactly one cell
            for ion in ions:
                lits = [var_a(R, 0, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
                enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                for cl in enc.clauses:
                    add_clause(cl)

            # ------------------------------------------------------------------
            # (1) Initial layout and inter-round chaining
            # ------------------------------------------------------------------

            # (1a) Initial layout: a[0,0] equals A_in
            for krow in range(n):
                for jcol in range(m):
                    ion0 = int(A_in[krow, jcol])
                    # Ion ion0 must be there
                    add_clause([var_a(0, 0, krow, jcol, ion0)])
                    # No other ion may be there at (0,0)
                    for ion in ions:
                        if ion != ion0:
                            add_clause([-var_a(0, 0, krow, jcol, ion)])

            # (1b) Chaining: a[r+1,0] <-> a[r,P_bound] for r=0..R-1
            for r in range(R):
                for krow in range(n):
                    for jcol in range(m):
                        for ion in ions:
                            a_next = var_a(r + 1, 0, krow, jcol, ion)
                            a_end = var_a(r, P_bound, krow, jcol, ion)
                            add_clause([-a_next, a_end])
                            add_clause([-a_end, a_next])

            # ------------------------------------------------------------------
            # (2) BT pinning at end-of-round (FINAL STATE ONLY)
            # ------------------------------------------------------------------

            for r in range(R):
                for ion in ions:
                    if is_reserved(r, ion):
                        d_fix, c_fix = BT[r][ion]
                        # Ion must occupy (d_fix,c_fix) at end-of-round r
                        add_clause([var_a(r, P_bound, d_fix, c_fix, ion)])
                        # Ion cannot be anywhere else at end-of-round r
                        for krow in range(n):
                            for jcol in range(m):
                                if (krow, jcol) != (d_fix, c_fix):
                                    add_clause([-var_a(r, P_bound, krow, jcol, ion)])

            # ------------------------------------------------------------------
            # (3) Phase structure: horizontal -> vertical
            # ------------------------------------------------------------------

            # (3a) Monotonicity: phase[p] -> phase[p+1], unless debug_allow_phase_flips
            if not debug_allow_phase_flips and P_bound > 1:
                for p in range(P_bound - 1):
                    add_clause([-var_phase(p), var_phase(p + 1)])

            # ------------------------------------------------------------------
            # (4) Horizontal and (5) Vertical comparators + semantics + copy constraints
            # ------------------------------------------------------------------

            for r in range(R):
                for p in range(P_bound):
                    phase_p = var_phase(p)

                    # ---------------- Horizontal comparators (row-wise) ----------------
                    for krow in range(n):
                        for jcol in range(m - 1):
                            sh = var_s_h(r, p, krow, jcol)

                            # Gating: if phase[p] = 1 (vertical), horizontal comparators must be off
                            # phase[p] = 1 ⇒ ¬s_h  ==  (¬phase[p] ∨ ¬s_h)
                            add_clause([-phase_p, -sh])

                            # Parity (odd-even) for horizontal comparisons:
                            # here we just use p%2 as horizontal parity:
                            #   if jcol % 2 != p % 2 then s_h must be 0
                            if jcol % 2 != (p % 2):
                                add_clause([-sh])

                            # Forward-only SWAP semantics between (krow,jcol) and (krow,jcol+1)
                            # NOTE: we ONLY constrain the s=1 (swap) case here.
                            #       The s=0 (identity) case is handled by the horizontal copy
                            #       constraints below, which are guarded on ¬phase and ¬s_left/¬s_right.
                            for ion in ions:
                                a_cur_j  = var_a(r, p,     krow, jcol,     ion)
                                a_cur_j1 = var_a(r, p,     krow, jcol + 1, ion)
                                a_next_j  = var_a(r, p + 1, krow, jcol,     ion)
                                a_next_j1 = var_a(r, p + 1, krow, jcol + 1, ion)

                                # (H3) If s=1 and ion is in right cell at p, it must be in left at p+1:
                                #      (s ∧ a_cur_j1) -> a_next_j
                                add_clause([-sh, -a_cur_j1, a_next_j])

                                # (H4) If s=1 and ion is in left cell at p, it must be in right at p+1:
                                #      (s ∧ a_cur_j) -> a_next_j1
                                add_clause([-sh, -a_cur_j,  a_next_j1])

                    # ---------------- Vertical comparators (column-wise) ----------------
                    for krow in range(n - 1):
                        for jcol in range(m):
                            sv = var_s_v(r, p, krow, jcol)

                            # Gating: if phase[p] = 0 (horizontal), vertical comparators must be off
                            # phase[p] = 0 ⇒ ¬s_v  ==  (phase[p] ∨ ¬s_v)
                            add_clause([phase_p, -sv])

                            # Parity (odd-even) for vertical comparisons, using p%2 as v-index:
                            #   if krow % 2 != p % 2 then s_v must be 0
                            if krow % 2 != (p % 2):
                                add_clause([-sv])

                            # Forward-only SWAP semantics between (krow,jcol) and (krow+1,jcol)
                            # Again, ONLY the s=1 case is constrained here; s=0 is enforced by
                            # the vertical copy constraints (phase ∧ ¬s_up ∧ ¬s_down).
                            for ion in ions:
                                a_cur_top  = var_a(r, p,     krow,     jcol, ion)
                                a_cur_bot  = var_a(r, p,     krow + 1, jcol, ion)
                                a_next_top = var_a(r, p + 1, krow,     jcol, ion)
                                a_next_bot = var_a(r, p + 1, krow + 1, jcol, ion)

                                # (V3) If s=1 and ion is in bottom cell at p, it must be in top at p+1:
                                #      (s ∧ a_cur_bot) -> a_next_top
                                add_clause([-sv, -a_cur_bot, a_next_top])

                                # (V4) If s=1 and ion is in top cell at p, it must be in bottom at p+1:
                                #      (s ∧ a_cur_top) -> a_next_bot
                                add_clause([-sv, -a_cur_top, a_next_bot])

            # -------- Copy constraints for non-participating cells (horizontal phase) --------
            #
            # For each (r,p,k,j,i), if phase[p]=0 (horizontal) AND no horizontal comparator
            # incident on (k,j) is active (i.e., s_h left/right both false), then:
            #   a[r,p,k,j,i] ↔ a[r,p+1,k,j,i]
            #
            # Implemented as implications guarded by (¬phase[p] ∧ ¬s_left ∧ ¬s_right).
            # This includes endpoints with s_h=0; if a comparator is active (s_h=1),
            # these copy constraints are disabled and the swap semantics above apply.

            for r in range(R):
                for p in range(P_bound):
                    phase_p = var_phase(p)
                    for krow in range(n):
                        for jcol in range(m):
                            # horizontal comparators that touch (krow,jcol)
                            s_left  = var_s_h(r, p, krow, jcol - 1) if jcol > 0     else None
                            s_right = var_s_h(r, p, krow, jcol)     if jcol < m - 1 else None

                            # antecedent: (¬phase[p] ∧ ¬s_left ∧ ¬s_right)
                            lits_ante_neg = [phase_p]  # this is ¬(¬phase), used on the clause side

                            if s_left is not None:
                                lits_ante_neg.append(s_left)
                            if s_right is not None:
                                lits_ante_neg.append(s_right)

                            for ion in ions:
                                a_cur  = var_a(r, p,     krow, jcol, ion)
                                a_next = var_a(r, p + 1, krow, jcol, ion)

                                # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_cur) -> a_next
                                # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_cur ∨ a_next)
                                add_clause(lits_ante_neg + [-a_cur, a_next])

                                # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_next) -> a_cur
                                # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_next ∨ a_cur)
                                add_clause(lits_ante_neg + [-a_next, a_cur])

            # -------- Copy constraints for non-participating cells (vertical phase) --------
            #
            # For each (r,p,k,j,i), if phase[p]=1 (vertical) AND no vertical comparator
            # incident on (k,j) is active (i.e., s_v up/down both false), then:
            #   a[r,p,k,j,i] ↔ a[r,p+1,k,j,i]
            #
            # Implemented as implications guarded by (phase[p] ∧ ¬s_up ∧ ¬s_down).
            # As above, this includes endpoints with s_v=0; if a comparator is active
            # at this cell, these copy constraints are disabled and the vertical swap
            # semantics take over.

            for r in range(R):
                for p in range(P_bound):
                    phase_p = var_phase(p)
                    for krow in range(n):
                        for jcol in range(m):
                            # vertical comparators that touch (krow,jcol)
                            s_up   = var_s_v(r, p, krow - 1, jcol) if krow > 0     else None
                            s_down = var_s_v(r, p, krow,     jcol) if krow < n - 1 else None

                            # antecedent: (phase[p] ∧ ¬s_up ∧ ¬s_down)
                            lits_ante_neg = [-phase_p]  # this is ¬phase on the clause side

                            if s_up is not None:
                                lits_ante_neg.append(s_up)
                            if s_down is not None:
                                lits_ante_neg.append(s_down)

                            for ion in ions:
                                a_cur  = var_a(r, p,     krow, jcol, ion)
                                a_next = var_a(r, p + 1, krow, jcol, ion)

                                # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_cur) -> a_next
                                # CNF: (¬phase ∨ s_up ∨ s_down ∨ ¬a_cur ∨ a_next)
                                add_clause(lits_ante_neg + [-a_cur, a_next])

                                # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_next) -> a_cur
                                # CNF: (¬phase ∨ s_up ∨ s_down ∨ ¬a_next ∨ a_cur)
                                add_clause(lits_ante_neg + [-a_next, a_cur])

            # ------------------------------------------------------------------
            # (5) End-of-round row/block abstraction and pair constraints
            # ------------------------------------------------------------------

            for r in range(R):
                # row_end / w_end linkage from final layout a[r,P_bound]
                for ion in ions:
                    # row_end
                    for d in range(n):
                        re = var_row_end(r, ion, d)
                        cell_lits = [var_a(r, P_bound, d, j, ion) for j in range(m)]
                        # row_end -> OR a
                        add_clause([-re] + cell_lits)
                        # each cell -> row_end
                        for aj in cell_lits:
                            add_clause([-aj, re])

                    # w_end
                    for b in range(num_blocks):
                        we = var_w_end(r, ion, b)
                        cells = []
                        for d in range(n):
                            for j in range(b * k, min((b + 1) * k, m)):
                                cells.append(var_a(r, P_bound, d, j, ion))
                        if cells:
                            add_clause([-we] + cells)
                            for aj in cells:
                                add_clause([-aj, we])
                        else:
                            # Block has no columns (edge case), force w_end false
                            add_clause([-we])

                # Pairs must share same row and block (unless skipping for debug)
                if not debug_skip_pair_constraints:
                    for (i1, i2) in P_arr[r]:
                        if i1 not in ions or i2 not in ions:
                            continue
                        # Same row
                        for d in range(n):
                            re1 = var_row_end(r, i1, d)
                            re2 = var_row_end(r, i2, d)
                            add_clause([-re1, re2])
                            add_clause([-re2, re1])
                        # Same block
                        for b in range(num_blocks):
                            we1 = var_w_end(r, i1, b)
                            we2 = var_w_end(r, i2, b)
                            add_clause([-we1, we2])
                            add_clause([-we2, we1])

            # ------------------------------------------------------------------
            # (6) Optional Level-3 soft clauses (boundary avoidance, swap-cost)
            # ------------------------------------------------------------------
            if use_wcnf and add_boundary_soft and (wB_row > 0 or wB_col > 0):
                # Precompute unique boundary cells for this subgrid:
                #   - last column (all rows)
                #   - last row (all columns)
                boundary_cells = set()
                # last column
                for d in range(n):
                    boundary_cells.add((d, m - 1))
                # last row
                for jcol in range(m):
                    boundary_cells.add((n - 1, jcol))
                boundary_cells = sorted(boundary_cells)

                # Limit rounds we add softs for (helps MaxSAT scaling)
                rounds_soft_cap = min(R, 2)

                for r in range(rounds_soft_cap):
                    for ion in ions:
                        in_minor = ion_in_minor_P_arr(r, ion)   # gate in this subgrid
                        in_full  = ion_in_full_P_arr(r, ion)    # gate somewhere (this or later)

                        # If ion never participates in any gate, don't care about its boundary placement.
                        if not in_minor and not in_full:
                            continue

                        # --------------------------------------------------------------
                        # Case 1: ion has a gate in this subgrid (current rounds)
                        #         → avoid putting it on the boundary at end-of-round.
                        # --------------------------------------------------------------
                        if in_minor:
                            lits_avoid = []

                            if wB_col > 0:
                                # Avoid last-column cells.
                                for d, jcol in boundary_cells:
                                    if jcol == m - 1:
                                        # Cost if ion is TRUE in a boundary cell:
                                        # clause is OR of negative literals, so violation = ion at boundary.
                                        lits_avoid.append(-var_a(r, P_bound, d, jcol, ion))

                            if wB_row > 0:
                                # Avoid last-row cells.
                                for d, jcol in boundary_cells:
                                    if d == n - 1:
                                        lits_avoid.append(-var_a(r, P_bound, d, jcol, ion))

                            if lits_avoid:
                                # Clause satisfied (no cost) if ion stays out of all boundary cells.
                                # Violated (cost) if ion is placed on any boundary cell.
                                add_soft_clause(lits_avoid, weight=max(wB_row, wB_col))

                        # --------------------------------------------------------------
                        # Case 2: ion has no gate in this subgrid but has gates elsewhere
                        #         → mildly attract it toward the boundary for future routing.
                        # --------------------------------------------------------------
                        elif in_full:
                            lits_pref = [var_a(r, P_bound, d, jcol, ion)
                                        for (d, jcol) in boundary_cells]

                            if lits_pref:
                                # Clause satisfied (no cost) if ion is TRUE in *some* boundary cell.
                                # Violated (cost) if ion is interior-only.
                                add_soft_clause(lits_pref, weight=min(wB_row, wB_col))


                # Swap-cost minimisation: penalise s_h and s_v being True
                # swap_weight = 1
                # if "wSwap" in globals():
                #     swap_weight = globals()["wSwap"]
                # if "wSwap" in locals():
                #     swap_weight = locals()["wSwap"]

                # for r in range(R):
                #     for p in range(P_bound):
                #         for krow in range(n):
                #             for jcol in range(m - 1):
                #                 add_soft_clause([-var_s_h(r, p, krow, jcol)], weight=swap_weight)
                #         for krow in range(n - 1):
                #             for jcol in range(m):
                #                 add_soft_clause([-var_s_v(r, p, krow, jcol)], weight=swap_weight)

   
 

            
            # ------------------------------------------------------------------
            # (7) Frozen-core constraints for subgrid growth (optional)
            # ------------------------------------------------------------------
            if freeze_params is not None:
                mode = freeze_params.get("mode", FREEZE_NONE)
                if mode != FREEZE_NONE:
                    n_core = min(freeze_params.get("n_core", 0), n)
                    m_core = min(freeze_params.get("m_core", 0), m)
                    P_freeze = min(freeze_params.get("P_freeze", 0), P_bound)
                    P_freeze_max = freeze_params.get("P_freeze_max", 0)
                    prev_vpool = freeze_params.get("prev_vpool", None)
                    prev_true_set = freeze_params.get("prev_true_set", None)

                    if prev_vpool is not None and prev_true_set is not None:
                        def old_truth(obj: Tuple[Any, ...]):
                            """
                            Return True/False/None for the variable 'obj' in the *old* model:
                            - True  if it existed and was True,
                            - False if it existed and was False,
                            - None  if it did not exist in the old formula.
                            """
                            old_vid = prev_vpool.obj2id.get(obj, None)
                            if old_vid is None:
                                return None
                            return (old_vid in prev_true_set)

                        # --- Freeze 'a' only where they were True ---
                        for r in range(R):
                            for p in range(P_freeze + 1):
                                for krow in range(n_core):
                                    for jcol in range(m_core):
                                        for ion in ions:
                                            old_val = old_truth(("a", r, p, krow, jcol, ion))
                                            if old_val is True:
                                                # Pin this ion at this cell/time
                                                add_clause([var_a(r, p, krow, jcol, ion)])
                                            # if old_val is False or None, do nothing; 
                                            # cardinalities + other pins will constrain them.
                            # if P_freeze_max>0 and P_freeze_max>P_freeze:
                            #     for krow in range(n_core):
                            #         for jcol in range(m_core):
                            #             for ion in ions:
                            #                 old_val = old_truth(("a", r, P_freeze_max, krow, jcol, ion))
                            #                 if old_val is True:
                            #                     # Pin this ion at this cell/time
                            #                     add_clause([var_a(r, p, krow, jcol, ion)])
                            #                 # if old_val is False or None, do nothing; 
                            #                 # cardinalities + other pins will constrain them.
                        

                        if mode == FREEZE_FULL:
                            # Freeze horizontal comparators in core for passes p < P_freeze
                            for r in range(R):
                                for p in range(P_freeze):
                                    for krow in range(n_core):
                                        for jcol in range(max(0, m_core - 1)):
                                            old_val = old_truth(("s_h", r, p, krow, jcol))
                                            if old_val is None:
                                                continue
                                            new_var = var_s_h(r, p, krow, jcol)
                                            if old_val:
                                                add_clause([new_var])
                                       

                            # Freeze vertical comparators in core for passes p < P_freeze
                            for r in range(R):
                                for p in range(P_freeze):
                                    for krow in range(max(0, n_core - 1)):
                                        for jcol in range(m_core):
                                            old_val = old_truth(("s_v", r, p, krow, jcol))
                                            if old_val is None:
                                                continue
                                            new_var = var_s_v(r, p, krow, jcol)
                                            if old_val:
                                                add_clause([new_var])
                               

                            # Freeze phase[p] bits for p < P_freeze
                            for p in range(P_freeze):
                                old_val = old_truth(("phase", p))
                                if old_val is None:
                                    continue
                                new_phase = var_phase(p)
                                if old_val:
                                    add_clause([new_phase])
                            



            # if DEBUG_DIAG_DETAILED and use_wcnf:
            #     print(
            #         f"[WISE] build_structural_cnf({phase_label}) stats: "
            #         f"vars={vpool.top}, hard={len(formula.hard)}, soft={len(formula.soft)}",
            #         flush=True,
            #     )
            # elif DEBUG_DIAG_DETAILED and not use_wcnf:
            #     print(
            #         f"[WISE] build_structural_cnf({phase_label}) stats: "
            #         f"vars={vpool.top}, hard={len(formula.clauses)}, soft=0",
            #         flush=True,
            #     )

            return formula, vpool, ions, var_a
     






        def run_wise_debug_suite():
            """
            Run a compact but comprehensive suite of debug checks for the WISE CNF builder.
            Assumes globals are available:
                A_in, P_arr, BT, n, m, k, R, ions, num_blocks,
                wB_row, wB_col, DEBUG_DIAG,
                _build_structural_cnf
            """

            if not DEBUG_DIAG:
                return

            print("\n========== WISE DEBUG SUITE (UPDATED FOR CELL-BASED s_v) ==========\n")

            # ---------------------- small helpers ----------------------

            def build_no_pair_cnf(P_debug: int, label: str, allow_phase_flips=False):
                """Call builder with pair constraints disabled (for base structural tests)."""
                nonlocal P_arr
                P_saved = P_arr
                P_arr = [[] for _ in range(R)]
                cnf, vpool, ions_loc, var_a = _build_structural_cnf(
                    P_debug,
                    use_wcnf=False,
                    add_boundary_soft=False,
                    phase_label=label,
                    debug_skip_pair_constraints=True,
                    debug_allow_phase_flips=allow_phase_flips,
                )
                P_arr = P_saved
                return cnf, vpool, ions_loc, var_a

            def var_s_h_of(vpool, r, p, krow, jcol):
                return vpool.id(("s_h", r, p, krow, jcol))

            def var_s_v_of(vpool, r, p, krow, jcol):
                return vpool.id(("s_v", r, p, krow, jcol))

            def var_phase_of(vpool, p):
                return vpool.id(("phase", p))

            def var_row_end_of(vpool, r, ion, d):
                return vpool.id(("row_end", r, ion, d))

            def var_w_end_of(vpool, r, ion, b):
                return vpool.id(("w_end", r, ion, b))

            # 1) Check: is there any model with at least one comparator TRUE? (no pairs)
            # ------------------------------------------------------------------------
            def test_any_comparator_active(P_debug=20):
                print("\n[DEBUG] ===== Checking if any comparator can be active (NO PAIRS) =====")
                cnf_cmp, vpool_cmp, ions_cmp, _ = build_no_pair_cnf(P_debug, "debug-any-comparator")

                solver_any = Solver(name="glucose4")
                solver_any.append_formula(cnf_cmp.clauses)

                any_cmp_clause = []
                for r in range(R):
                    for p in range(P_debug):
                        for krow in range(n):
                            for jcol in range(m - 1):
                                any_cmp_clause.append(var_s_h_of(vpool_cmp, r, p, krow, jcol))
                        for krow in range(n - 1):
                            for jcol in range(m):
                                any_cmp_clause.append(var_s_v_of(vpool_cmp, r, p, krow, jcol))

                if not any_cmp_clause:
                    print("[DEBUG] No comparator vars exist at all (unexpected).")
                    solver_any.delete()
                    return

                solver_any.add_clause(any_cmp_clause)
                sat_any = solver_any.solve()
                print(f"[DEBUG] Exists model with at least one comparator TRUE? {sat_any}")
                if sat_any:
                    model = solver_any.get_model()
                    mset = {l for l in model if l > 0}
                    def true(lit): return lit in mset
                    print("[DEBUG] Active comparators in one such model:")
                    for r in range(R):
                        for p in range(P_debug):
                            for krow in range(n):
                                for jcol in range(m - 1):
                                    v = var_s_h_of(vpool_cmp, r, p, krow, jcol)
                                    if true(v):
                                        print(f"    s_h[r={r}, p={p}, k={krow}, j={jcol}] = True")
                            for krow in range(n - 1):
                                for jcol in range(m):
                                    v = var_s_v_of(vpool_cmp, r, p, krow, jcol)
                                    if true(v):
                                        print(f"    s_v[r={r}, p={p}, k={krow}, j={jcol}] = True")

                solver_any.delete()

            # 2) Check: forcing all passes horizontal / all vertical
            # ------------------------------------------------------
            def test_phase_force(P_debug=20):
                print("\n[DEBUG] ===== Forcing all passes horizontal / vertical (NO PAIRS) =====")

                cnf_phase, vpool_phase, _, _ = build_no_pair_cnf(
                    P_debug, "debug-phase-force", allow_phase_flips=False
                )

                def sat_with_units(units, label):
                    s = Solver(name="glucose4")
                    s.append_formula(cnf_phase.clauses)
                    for lit in units:
                        s.add_clause([lit])
                    ok = s.solve()
                    print(f"[DEBUG] {label}: SAT? {ok}")
                    s.delete()

                # all passes horizontal: phase[p] = 0 => ¬phase[p]
                units_all_h = [-var_phase_of(vpool_phase, p) for p in range(P_debug)]
                sat_with_units(units_all_h, "all passes horizontal (phase=0)")

                # all passes vertical: phase[p] = 1 => phase[p]
                units_all_v = [var_phase_of(vpool_phase, p) for p in range(P_debug)]
                sat_with_units(units_all_v, "all passes vertical (phase=1)")

            # 3) Force some comparator active and inspect layout (r=0)
            # --------------------------------------------------------
            def test_force_some_comparator(P_debug=5):
                print("\n[DEBUG] ===== Forcing some comparator active and inspecting layout =====")

                cnf_move, vpool_move, ions_move, var_a_move = build_no_pair_cnf(
                    P_debug, "debug-force-move"
                )

                solver_mv = Solver(name="glucose4")
                solver_mv.append_formula(cnf_move.clauses)

                at_least_one = []
                for r in range(R):
                    for p in range(P_debug):
                        for krow in range(n):
                            for jcol in range(m - 1):
                                at_least_one.append(var_s_h_of(vpool_move, r, p, krow, jcol))
                        for krow in range(n - 1):
                            for jcol in range(m):
                                at_least_one.append(var_s_v_of(vpool_move, r, p, krow, jcol))

                if at_least_one:
                    solver_mv.add_clause(at_least_one)

                sat_mv = solver_mv.solve()
                print(f"[DEBUG] base+\"some comparator\" SAT? {sat_mv}")
                if sat_mv:
                    model = solver_mv.get_model()
                    mset = {l for l in model if l > 0}
                    def true(l): return l in mset

                    def layout_at(p):
                        grid = np.empty_like(A_in)
                        for d in range(n):
                            for c in range(m):
                                found = None
                                for ion in ions_move:
                                    if true(var_a_move(0, p, d, c, ion)):
                                        found = ion
                                        break
                                grid[d, c] = -1 if found is None else found
                        return grid

                    print("[DEBUG] layout at p=0:")
                    print(layout_at(0))
                    print("[DEBUG] layout at p=P_debug:")
                    print(layout_at(P_debug))

                    print("[DEBUG] comparators TRUE in this model:")
                    for r in range(R):
                        for p in range(P_debug):
                            for krow in range(n):
                                for jcol in range(m - 1):
                                    v = var_s_h_of(vpool_move, r, p, krow, jcol)
                                    if true(v):
                                        print(f"    s_h[r={r}, p={p}, k={krow}, j={jcol}] = True")
                            for krow in range(n - 1):
                                for jcol in range(m):
                                    v = var_s_v_of(vpool_move, r, p, krow, jcol)
                                    if true(v):
                                        print(f"    s_v[r={r}, p={p}, k={krow}, j={jcol}] = True")

                solver_mv.delete()

            # 4) Enumerate reachable end cells for each ion (no pairs)
            # --------------------------------------------------------
            def test_reachability_per_ion(P_debug=20):
                print("\n[DEBUG] ===== Enumerating reachable end cells per ion (NO PAIRS) =====")

                cnf_base, vpool_base, ions_base, var_a_base = build_no_pair_cnf(
                    P_debug, "debug-reachability"
                )

                print(f"[DEBUG] base CNF (no pairs): vars={vpool_base.top}, clauses={len(cnf_base.clauses)}")

                def check_reachable(ion, d, c):
                    s = Solver(name="glucose4")
                    s.append_formula(cnf_base.clauses)
                    s.add_clause([var_a_base(0, P_debug, d, c, ion)])
                    sat = s.solve()
                    s.delete()
                    return sat

                for ion in ions_base:
                    cells = []
                    for d in range(n):
                        for c in range(m):
                            if check_reachable(ion, d, c):
                                cells.append((d, c))
                    print(f"  [DEBUG] Ion {ion} reachable end cells: {cells}")

            # 5) Cross-block reachability for specific ions (using w_end)
            # -----------------------------------------------------------
            def test_cross_block_reach(P_debug=20, test_ions=(1, 4)):
                print("\n[DEBUG] ===== Testing cross-block reachability (NO PAIRS) =====")

                cnf_blocks, vpool_blocks, ions_blocks, var_a_blocks = build_no_pair_cnf(
                    P_debug, "debug-block-reach"
                )

                num_blocks_local = math.ceil(m / k)

                def block_of_col(j): return j // k

                def can_ion_reach_block(ion, b):
                    s = Solver(name="glucose4")
                    s.append_formula(cnf_blocks.clauses)
                    lits = []
                    for d in range(n):
                        for c in range(m):
                            if block_of_col(c) == b:
                                lits.append(var_a_blocks(0, P_debug, d, c, ion))
                    if not lits:
                        s.delete()
                        return False
                    s.add_clause(lits)
                    sat = s.solve()
                    s.delete()
                    return sat

                for ion in test_ions:
                    if ion not in ions_blocks:
                        continue
                    print(f"\n[DEBUG] Ion {ion}:")
                    for b in range(num_blocks_local):
                        print(f"    can reach block {b}? {can_ion_reach_block(ion, b)}")

            # 6) Horizontal-only mini-network sanity check (no vertical, no phase)
            # --------------------------------------------------------------------
            def test_horizontal_only_toy(P_debug=5):
                print("\n[DEBUG] ===== Horizontal-only mini-network sanity check =====")

                A0 = np.asarray(A_in, dtype=int)
                n0, m0 = A0.shape
                ions0 = sorted(int(x) for x in A0.flatten())
                num_blocks0 = math.ceil(m0 / k)

                vpool = IDPool()
                cnf = CNF()

                def var_a(p, krow, jcol, ion): return vpool.id(("aH", p, krow, jcol, ion))
                def var_s(p, krow, jcol):      return vpool.id(("sH", p, krow, jcol))
                def block_of_col(j):           return j // k

                # one-ion-per-cell at each p
                for p in range(P_debug + 1):
                    for d in range(n0):
                        for c in range(m0):
                            lits = [var_a(p, d, c, ion) for ion in ions0]
                            enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                            cnf.extend(enc.clauses)

                # initial layout
                for d in range(n0):
                    for c in range(m0):
                        ion0 = int(A0[d, c])
                        cnf.append([var_a(0, d, c, ion0)])

                # horizontal semantics
                for p in range(P_debug):
                    for d in range(n0):
                        for c in range(m0 - 1):
                            s = var_s(p, d, c)
                            if c % 2 != (p % 2):
                                cnf.append([-s])
                            for ion in ions0:
                                a_c = var_a(p, d, c, ion)
                                a_c1 = var_a(p, d, c + 1, ion)
                                a_n = var_a(p + 1, d, c, ion)
                                a_n1 = var_a(p + 1, d, c + 1, ion)

                                cnf.append([ s, -a_c,  a_n  ])
                                cnf.append([ s, -a_c1, a_n1 ])
                                cnf.append([-s, -a_c1, a_n  ])
                                cnf.append([-s, -a_c,  a_n1 ])

                # ask if ions (1,4) can end in same block
                pair = (1, 4)
                for b in range(num_blocks0):
                    print(f"\n[H-ONLY DEBUG] Trying to place ions {pair} both in block {b} ...")
                    s = Solver(name="glucose4")
                    s.append_formula(cnf.clauses)
                    for ion in pair:
                        lits = []
                        for d in range(n0):
                            for c in range(m0):
                                if block_of_col(c) == b:
                                    lits.append(var_a(P_debug, d, c, ion))
                        s.add_clause(lits)
                    sat = s.solve()
                    print(f"    SAT? {sat}")
                    if sat:
                        model = s.get_model()
                        mset = {l for l in model if l > 0}
                        def true(v): return v in mset
                        for ion in pair:
                            cells = []
                            for d in range(n0):
                                for c in range(m0):
                                    if true(var_a(P_debug, d, c, ion)):
                                        cells.append((d, c))
                            print(f"      ion {ion} final cells: {cells}")
                    s.delete()

            # 7) Base CNF SAT? then incremental: +phase monotonicity, +pair constraints
            # -------------------------------------------------------------------------
            def test_incremental_pairs(P_test=20):
                print("\n[DEBUG] ===== Incremental base / phase / pair constraints =====")

                # full builder but with pair constraints skipped
                cnf_base, vpool_b, ions_b, _ = _build_structural_cnf(
                    P_test,
                    use_wcnf=False,
                    add_boundary_soft=False,
                    phase_label="debug",
                    debug_skip_pair_constraints=True,
                    debug_allow_phase_flips=True,
                )

                s = Solver(name="glucose4")
                s.append_formula(cnf_base.clauses)
                ok = s.solve()
                print(f"After basic constraints (no pairs, free phase): SAT? {ok}")
                if not ok:
                    s.delete()
                    return

                # add monotonic phase
                for p in range(P_test - 1):
                    p0 = var_phase_of(vpool_b, p)
                    p1 = var_phase_of(vpool_b, p + 1)
                    s.add_clause([-p0, p1])
                ok = s.solve()
                print(f"After adding phase monotonicity: SAT? {ok}")
                if not ok:
                    s.delete()
                    return

                # add all pair constraints for round 0
                num_blocks_local = math.ceil(m / k)
                for (i1, i2) in P_arr[0]:
                    if i1 in ions_b and i2 in ions_b:
                        for d in range(n):
                            re1 = var_row_end_of(vpool_b, 0, i1, d)
                            re2 = var_row_end_of(vpool_b, 0, i2, d)
                            s.add_clause([-re1, re2])
                            s.add_clause([-re2, re1])
                        for b in range(num_blocks_local):
                            we1 = var_w_end_of(vpool_b, 0, i1, b)
                            we2 = var_w_end_of(vpool_b, 0, i2, b)
                            s.add_clause([-we1, we2])
                            s.add_clause([-we2, we1])
                ok = s.solve()
                print(f"After adding pair constraints for all pairs: SAT? {ok}")
                s.delete()

                # each pair alone
                cnf_base_single, vpool_single, ions_single, var_a_single = build_no_pair_cnf(
                    P_test, "debug-single-pair"
                )
                num_blocks_single = math.ceil(m / k)

                for (i1, i2) in P_arr[0]:
                    sp = Solver(name="glucose4")
                    sp.append_formula(cnf_base_single.clauses)
                    if i1 in ions_single and i2 in ions_single:
                        for d in range(n):
                            re1 = var_row_end_of(vpool_single, 0, i1, d)
                            re2 = var_row_end_of(vpool_single, 0, i2, d)
                            sp.add_clause([-re1, re2])
                            sp.add_clause([-re2, re1])
                        for b in range(num_blocks_single):
                            we1 = var_w_end_of(vpool_single, 0, i1, b)
                            we2 = var_w_end_of(vpool_single, 0, i2, b)
                            sp.add_clause([-we1, we2])
                            sp.add_clause([-we2, we1])
                    sat_single = sp.solve()
                    print(f"Pair {(i1, i2)} alone: SAT? {sat_single}")
                    sp.delete()

            # 8) Relaxed block-equality sanity check (no routing at all)
            # ---------------------------------------------------------
            def test_relaxed_block_equality(pair=(1, 4)):
                print("\n[DEBUG] ===== Relaxed block-equality sanity check =====")
                i1, i2 = pair
                k_block = k
                vpool = IDPool()
                ions_all = sorted(int(x) for x in A_in.flatten())
                num_blocks_local = math.ceil(m / k_block)
                P_bound = 0  # only p=0 (no routing)

                def var_a(p, krow, jcol, ion): return vpool.id(("arel", p, krow, jcol, ion))
                def var_w(ion, b):             return vpool.id(("wrel", ion, b))

                formula = CNF()

                # one-ion-per-cell at p=0 and p=P_bound (same here)
                for p in (0, P_bound):
                    for d in range(n):
                        for c in range(m):
                            lits = [var_a(p, d, c, ion) for ion in ions_all]
                            enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                            formula.extend(enc.clauses)

                # initial layout fixed at p=0
                for d in range(n):
                    for c in range(m):
                        ion0 = int(A_in[d, c])
                        formula.append([var_a(0, d, c, ion0)])

                # w_end def at p=P_bound
                for ion in ions_all:
                    for b in range(num_blocks_local):
                        we = var_w(ion, b)
                        cells = []
                        for d in range(n):
                            for j in range(b * k_block, min((b + 1) * k_block, m)):
                                cells.append(var_a(P_bound, d, j, ion))
                        if cells:
                            formula.append([-we] + cells)
                            for aj in cells:
                                formula.append([-aj, we])
                        else:
                            formula.append([-we])

                # block equality for the pair
                for b in range(num_blocks_local):
                    w1b = var_w(i1, b)
                    w2b = var_w(i2, b)
                    formula.append([-w1b, w2b])
                    formula.append([-w2b, w1b])

                s = Solver(name="glucose4")
                s.append_formula(formula.clauses)
                sat = s.solve()
                print(f"[DEBUG relaxed] pair={pair}, SAT? {sat}")
                s.delete()

            # 9) Base CNF with *full* spec (including pairs) SAT? + inspect layout
            # --------------------------------------------------------------------
            def test_full_base_cnf(P_debug=20):
                print("\n[DEBUG] ===== Full base CNF (with pairs) SAT check =====")
                cnf_full, vpool_full, ions_full, var_a_full = _build_structural_cnf(
                    P_debug,
                    use_wcnf=False,
                    add_boundary_soft=False,
                    phase_label="debug-full",
                )
                print(f"[DEBUG] base CNF: vars={vpool_full.top}, clauses={len(cnf_full.clauses)}")

                s = Solver(name="glucose4")
                s.append_formula(cnf_full.clauses)
                sat = s.solve()
                print("[DEBUG] base formula SAT? ", sat)
                if not sat:
                    s.delete()
                    return

                model = s.get_model()
                s.delete()
                mset = {l for l in model if l > 0}

                def true(v): return v in mset

                def block_of_col(j): return j // k
                num_blocks_local = math.ceil(m / k)

                # reconstruct layout for r=0, p=P_debug
                print("\n[DEBUG] End-of-round layout (r=0, p=P_debug):")
                grid = np.empty_like(A_in)
                for d in range(n):
                    for c in range(m):
                        found = None
                        for ion in ions_full:
                            if true(var_a_full(0, P_debug, d, c, ion)):
                                found = ion
                                break
                        grid[d, c] = -1 if found is None else found
                print(grid)

                print("\n[DEBUG] Initial layout (A_in) with blocks:")
                print(A_in)
                for d in range(n):
                    for c in range(m):
                        ion = int(A_in[d, c])
                        print(f"  ion {ion} starts at (row={d}, col={c}, block={block_of_col(c)})")

                print("\n[DEBUG] End-of-round block membership for each ion:")
                for ion in ions_full:
                    cells = []
                    for d in range(n):
                        for c in range(m):
                            if true(var_a_full(0, P_debug, d, c, ion)):
                                cells.append((d, c))
                    blocks_from_cells = sorted({block_of_col(c) for (_, c) in cells})

                    blocks_from_w = []
                    for b in range(num_blocks_local):
                        we = var_w_end_of(vpool_full, 0, ion, b)
                        if true(we):
                            blocks_from_w.append(b)

                    print(f"  ion {ion}:")
                    print(f"     final cells       : {cells}")
                    print(f"     blocks from cells: {blocks_from_cells}")
                    print(f"     blocks from w_end: {blocks_from_w}")

            # ---------------------- run all tests ----------------------

            test_any_comparator_active(P_debug=20)
            test_phase_force(P_debug=20)
            test_force_some_comparator(P_debug=5)
            test_reachability_per_ion(P_debug=20)
            test_cross_block_reach(P_debug=20, test_ions=(1, 4))
            test_horizontal_only_toy(P_debug=5)
            test_incremental_pairs(P_test=20)
            test_relaxed_block_equality(pair=(1, 4))
            test_full_base_cnf(P_debug=20)


            if DEBUG_DIAG:
                print("\n[DEBUG] ===== Single-comparator micro sanity check =====")

                def _debug_single_comparator_micro():
                    """
                    Tiny 1×2, 1-round, 1-pass network:

                        p=0: [1, 2]
                        p=1: ?

                    We encode:
                        - exactly-one per cell and per ion at p=0,1
                        - one horizontal comparator s_h(0,0,0,0)
                        - SAME swap + copy structure as in _build_structural_cnf,
                        but without phases or vertical comparators.

                    Then we check:
                        - SAT with s free,
                        - SAT with s=0,
                        - SAT with s=1 and inspect the resulting layout.
                    """
                    vpool = IDPool()
                    n, m = 1, 2
                    ions = [1, 2]
                    P_bound = 1  # one pass

                    def var_a(p, krow, jcol, ion):
                        return vpool.id(("a", p, krow, jcol, ion))

                    def var_s(p, krow, jcol):
                        return vpool.id(("s_h", p, krow, jcol))

                    cnf = CNF()

                    # (0) exactly-one per cell and per ion at p=0,1
                    for p in range(P_bound + 1):  # p=0,1
                        # per cell
                        for krow in range(n):
                            for jcol in range(m):
                                lits = [var_a(p, krow, jcol, ion) for ion in ions]
                                enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                                cnf.extend(enc.clauses)
                        # per ion
                        for ion in ions:
                            lits = [var_a(p, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
                            enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                            cnf.extend(enc.clauses)

                    # (1) Initial layout at p=0: [1, 2]
                    cnf.append([var_a(0, 0, 0, 1)])  # cell(0,0) = ion 1
                    cnf.append([var_a(0, 0, 1, 2)])  # cell(0,1) = ion 2

                    sh = var_s(0, 0, 0)

                    # (2) Horizontal semantics as in the main builder (after our fix):
                    for ion in ions:
                        a_cur_j  = var_a(0, 0, 0, ion)
                        a_cur_j1 = var_a(0, 0, 1, ion)
                        a_next_j  = var_a(1, 0, 0, ion)
                        a_next_j1 = var_a(1, 0, 1, ion)

                        # SWAP only for s=1:
                        # (s ∧ a_cur_j1) -> a_next_j
                        cnf.append([-sh, -a_cur_j1, a_next_j])
                        # (s ∧ a_cur_j)  -> a_next_j1
                        cnf.append([-sh, -a_cur_j,  a_next_j1])

                    # (3) Copy constraints for non-participating cells with "horizontal phase"
                    # In this micro test, there is only one comparator touching both cells, so:
                    #   s_left/s_right = sh appropriately.

                    p = 0
                    for krow in range(n):
                        for jcol in range(m):
                            s_left  = var_s(p, krow, jcol - 1) if jcol > 0     else None
                            s_right = var_s(p, krow, jcol)     if jcol < m - 1 else None

                            lits_ante_neg = []  # we assume "phase=0" so only ¬s_left/¬s_right matter
                            if s_left is not None:
                                lits_ante_neg.append(s_left)
                            if s_right is not None:
                                lits_ante_neg.append(s_right)

                            for ion in ions:
                                a_cur  = var_a(0, krow, jcol, ion)
                                a_next = var_a(1, krow, jcol, ion)

                                # (¬s_left ∧ ¬s_right ∧ a_cur) -> a_next
                                cnf.append(lits_ante_neg + [-a_cur, a_next])
                                # (¬s_left ∧ ¬s_right ∧ a_next) -> a_cur
                                cnf.append(lits_ante_neg + [-a_next, a_cur])

                    def solve_with(extra_units, label):
                        s = Solver(name="glucose4")
                        s.append_formula(cnf.clauses)
                        for u in extra_units:
                            s.add_clause([u])
                        sat = s.solve()
                        print(f"[DEBUG micro] {label}: SAT? {sat}")
                        if sat:
                            model = s.get_model()
                            s.delete()
                            mset = {l for l in model if l > 0}
                            def true(v): return v in mset

                            layout0 = [[None for _ in range(m)] for _ in range(n)]
                            layout1 = [[None for _ in range(m)] for _ in range(n)]
                            for ion in ions:
                                for krow in range(n):
                                    for jcol in range(m):
                                        if true(var_a(0, krow, jcol, ion)):
                                            layout0[krow][jcol] = ion
                                        if true(var_a(1, krow, jcol, ion)):
                                            layout1[krow][jcol] = ion
                            print(f"    p=0 layout: {layout0}")
                            print(f"    p=1 layout: {layout1}")
                            print(f"    s_h = {true(sh)}")
                        else:
                            s.delete()

                    # Run three scenarios:
                    solve_with([],         "s free")
                    solve_with([-sh],      "forced s=0")
                    solve_with([sh],       "forced s=1")

                _debug_single_comparator_micro()
                print("\n========== END WISE DEBUG SUITE ==========\n")

        def decode_wise_schedule_from_model(
            model: List[int],
            vpool,
            n: int,
            m: int,
            R: int,
            P_bound: int,
        ) -> List[Dict[str, Any]]:
            """
            Decode the SAT/RC2 model into a per-pass schedule of comparators.

            Returns:
                schedule: list of length P_bound, where each entry is a dict:
                    {
                        "phase": "H" or "V",
                        "h_swaps": [(row, col), ...],    # (krow, jcol) : swap (krow,jcol)<->(krow,jcol+1)
                        "v_swaps": [(row, col), ...],    # (krow, jcol) : swap (krow,jcol)<->(krow+1,jcol)
                    }
            """
            model_set = {lit for lit in model if lit > 0}

            def lit_true(v: int) -> bool:
                return v in model_set

            def var_s_h(r, p, krow, jcol):
                return vpool.id(("s_h", r, p, krow, jcol))

            def var_s_v(r, p, krow, jcol):
                return vpool.id(("s_v", r, p, krow, jcol))

            def var_phase(p):
                return vpool.id(("phase", p))

            schedule: List[Dict[str, Any]] = []

            for p in range(P_bound):
                phase_lit = var_phase(p)
                is_vertical = (phase_lit <= vpool.top and lit_true(phase_lit))
                phase = "V" if is_vertical else "H"

                pass_info: Dict[str, Any] = {
                    "phase": phase,
                    "h_swaps": [],
                    "v_swaps": [],
                }

                # We look at only the next round
                # TODO: we should probably just run MAXSAT once when doing full lookahead, or have a parameter that decides how many rounds we schedule
                r=0
                if phase == "H":
                    # Horizontal comparators at this pass
                    for krow in range(n):
                        for jcol in range(m - 1):
                            v = var_s_h(r, p, krow, jcol)
                            if v <= vpool.top and lit_true(v):
                                pass_info["h_swaps"].append((krow, jcol))
                else:
                    # Vertical comparators at this pass
                    for krow in range(n - 1):
                        for jcol in range(m):
                            v = var_s_v(r, p, krow, jcol)
                            if v <= vpool.top and lit_true(v):
                                pass_info["v_swaps"].append((krow, jcol))

                schedule.append(pass_info)

            return schedule

        # run_wise_debug_suite()
        

        if freeze_seed_prev is not None:
            n_core0 = freeze_seed_prev["n_core0"]
            m_core0 = freeze_seed_prev["m_core0"]
            prev_P_star = freeze_seed_prev["P_freeze0"]
            prev_vpool = freeze_seed_prev["prev_vpool"]
            prev_model = list(freeze_seed_prev["prev_true_set"])
        else:
            n_core0 = m_core0 = P_freeze0 = 0
            prev_vpool = None
            prev_model = None
            prev_P_star = 0
        prev_true_set: Optional[Set[int]] = None
        if prev_model is not None:
            prev_true_set = {lit for lit in prev_model if lit > 0}



        # For the very first subgrid we have no previous model → no freezing
        if freeze_seed_prev is None:
            n_core_max = 0
            m_core_max = 0
            P_freeze_max = 0
        else:
            # Clamp core and P_freeze against current geometry / P_max
            n_core_max = min(n_core0, n)
            m_core_max = min(m_core0, m)
            # Only sensible to freeze passes we know something about; we cap by prev_P_star
            if prev_P_star is not None:
                P_freeze_max = prev_P_star
            else:
                P_freeze_max = 0

        P_max=max(n+m, prev_P_star+max(n,m))

        isDecCols = True
        # -------------------------------
        # Level 1.5: Adaptive core freezing across growing subgrids
        # -------------------------------
        def _enumerate_freeze_configs(
                P_step_size: int = 1,
                n_step_size: int = 1,
                m_step_size: int = 1
        ) -> Iterable[Tuple[int, int, int]]:
            """
            Yield (n_c, m_c, P_freeze) in descending 'freeze volume' order,
            where freeze volume = n_c * m_c * (P_freeze).

            This ensures that the first SAT config we accept uses the largest
            frozen core and the deepest time freeze consistent with the instance.
            """
            configs = []
            for n_c in range(n_core_max, -1, -n_step_size):
                for m_c in range(m_core_max, -1, -m_step_size):
                    # we allow P_freeze = 0..P_freeze_max
                    for P_freeze in range(P_freeze_max, -1, -P_step_size):
                        vol = n_c * m_c * (P_freeze)
                        configs.append((vol, n_c, m_c, P_freeze))

            # Sort descending by volume, then tie-break by n_c, m_c, P_freeze
            configs.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)

            for _, n_c, m_c, P_freeze in configs:
                yield n_c, m_c, P_freeze

        chosen_solution=None
        for n_c, m_c, P_freeze in _enumerate_freeze_configs(P_step_size=max(int(P_freeze_max/2),1), m_step_size=max(int(m_core_max/2),1), n_step_size=max(int(n_core_max/2),1)):
            mode = FREEZE_POS_ONLY
            freeze_params = {
                "n_core": n_c,
                "m_core": m_c,
                "P_freeze": P_freeze,
                "P_freeze_max": P_freeze_max,
                "mode": mode,
                "prev_vpool": prev_vpool,
                "prev_true_set": prev_true_set,
            }

            # -------------------------------
            # Level 2: Binary search on P (minimum number of passes per round)
            # -------------------------------
            P_lo = max(0, prev_P_star-1)
            P_hi =P_max # upper bound on passes (e.g., worst-case moves across n rows + m columns)
            P_star = None

            if DEBUG_DIAG:
                print(f"[WISE] starting binary search for P in [0, {P_hi}] with n_c={n_c}, m_c={m_c}, P_freeze={P_freeze} and mode={FREEZE_POS_ONLY}", flush=True)

            while P_lo <= P_hi:
                P_mid = (P_lo + P_hi) // 2

                cnf_mid, vpool_mid, ions_mid, var_a_mid = _build_structural_cnf(
                    P_mid,
                    use_wcnf=False,
                    add_boundary_soft=False,
                    phase_label=f"P={P_mid}/SAT",
                    freeze_params=freeze_params
                )

                t_sat_start = time.time()
                sat_ok, model_mid, status_sat = run_sat_with_timeout_file(
                    cnf_mid,
                    timeout_s=max_sat_time,
                    debug_prefix=None,
                )

                # with Minisat22(bootstrap_with=cnf_mid.clauses) as sat:
                #     sat_ok = sat.solve()
                #     model_mid = sat.get_model() if sat_ok else None
                #     status_sat = "ok" if sat_ok else "error"
                t_sat_end = time.time()

                if DEBUG_DIAG:
                    if hasattr(cnf_mid, "clauses"):
                        n_clauses = len(cnf_mid.clauses)
                    else:
                        n_clauses = len(cnf_mid.hard)  # just in case
                 
                if sat_ok:
                    print(
                        f"[WISE]  test P={P_mid}, n_c={n_c}, m_c={m_c},: status={status_sat}, SAT={sat_ok}, "
                        f"vars={vpool_mid.top}, clauses={n_clauses}, "
                        f"time={t_sat_end - t_sat_start:.3f}s",
                        flush=True,
                    )
                    chosen_solution = (cnf_mid, vpool_mid, ions_mid, var_a_mid, model_mid,
                                    freeze_params)
                    # Record this as the best so far and tighten upper bound
                    P_star = P_mid
                    P_hi = P_mid - 1
                else:
                    # UNSAT: need more passes
                    P_lo = P_mid + 1

        

                # if status_sat != "ok":
                #     # timeout or error
                #     if best_P is not None:
                #         if DEBUG_DIAG:
                #             print(
                #                 f"[WISE] SAT at P={P_mid} ended with status={status_sat}; "
                #                 f"falling back to previous best P={best_P[0]} and stopping binary search.",
                #                 flush=True,
                #             )
                #         break
                #     else:
                #         raise RuntimeError(
                #             f"SAT solver failed with status={status_sat} before any feasible P was found."
                #         )
            
            if (P_star is not None) or (n_c==0 or m_c==0):
                break
            if isDecCols:
                m_c -=1
                isDecCols = (n_c==1)
            else:
                n_c -=1
                isDecCols = (m_c>1)

        

        if P_star is None:
            raise RuntimeError("No feasible layout for any P in [0, n+m].")

        _, vpool_sat, ions_sat, var_a_sat, sat_model_star, freeze_used = chosen_solution
        if DEBUG_DIAG:
            print(f"[WISE] minimal P* found: {P_star}", flush=True)

        # -------------------------------
        # Level 3: MaxSAT refinement at P*
        # -------------------------------
        if DEBUG_DIAG:
            print(f"[WISE] building WCNF at P*={P_star} for MaxSAT...", flush=True)

        t_build_start = time.time()
        wcnf, vpool_w, ions_w, var_a_w = _build_structural_cnf(
            P_star,
            use_wcnf=True,
            add_boundary_soft=True,
            phase_label=f"P*={P_star}/WCNF",
            freeze_params=freeze_used
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
        # rc2 = RC2(wcnf)
        # model_rc2 = rc2.compute()
        # cost_rc2 = rc2.cost if model_rc2 is not None else None
        # status_rc2 = "ok" if model_rc2 is not None else "error"

        if DEBUG_DIAG:
            print(
                f"[WISE] RC2 status={status_rc2}, opt_cost={cost_rc2}",
                flush=True,
            )

        # if DEBUG_DIAG:
        #     print("\n[DEBUG] ===== Decoding used passes from RC2 model =====")

        #     # model: list[int] from RC2
        #     model_set = {lit for lit in model_rc2 if lit > 0}

        #     def lit_true(v: int) -> bool:
        #         return v in model_set

        #     # Helpers must match the IDs used in _build_structural_cnf
        #     def var_s_h_dec(r, p, krow, jcol):
        #         return vpool_w.id(("s_h", r, p, krow, jcol))

        #     def var_s_v_dec(r, p, krow, jcol):
        #         return vpool_w.id(("s_v", r, p, krow, jcol))

            # def var_phase_dec(p):
            #     return vpool_w.id(("phase", p))

            # max_p_with_cmp = -1
            # max_p_with_h = -1
            # max_p_with_v = -1

            # for r in range(R):
            #     for p in range(P_star):        # note: 0..P_bound-1
            #         any_h = False
            #         any_v = False

            #         # horizontal comparators at this pass
            #         for krow in range(n):
            #             for jcol in range(m - 1):
            #                 v = var_s_h_dec(r, p, krow, jcol)
            #                 if v <= vpool_w.top and lit_true(v):
            #                     any_h = True

        #             # vertical comparators at this pass
        #             for krow in range(n - 1):
        #                 for jcol in range(m):
        #                     v = var_s_v_dec(r, p, krow, jcol)
        #                     if v <= vpool_w.top and lit_true(v):
        #                         any_v = True

        #             if any_h or any_v:
        #                 max_p_with_cmp = max(max_p_with_cmp, p)
        #                 if any_h:
        #                     max_p_with_h = max(max_p_with_h, p)
        #                 if any_v:
        #                     max_p_with_v = max(max_p_with_v, p)

        #     print(f"[DEBUG] P_bound (P*)   = {P_star}")
        #     print(f"[DEBUG] max p with any comparator TRUE = {max_p_with_cmp}")
        #     print(f"[DEBUG] max p with H comparator TRUE   = {max_p_with_h}")
        #     print(f"[DEBUG] max p with V comparator TRUE   = {max_p_with_v}")

        #     # Also dump phase pattern for clarity
        #     phase_vals = []
        #     for p in range(P_star):
        #         v = var_phase_dec(p)
        #         val = None
        #         if v <= vpool_w.top:
        #             val = 1 if lit_true(v) else 0
        #         phase_vals.append(val)
        #     print(f"[DEBUG] phase[p] for p=0..{P_star-1}: {phase_vals}")
        # # Decide which model to use
        if status_rc2 == "ok" and model_rc2 is not None:
            model_used = model_rc2
            vpool_used = vpool_w
            var_a_used = var_a_w
            ions_used = ions_w
            if DEBUG_DIAG:
                print("[WISE] using RC2 MaxSAT model at P*", flush=True)
        else:
            model_used = sat_model_star
            vpool_used = vpool_sat
            var_a_used = var_a_sat
            ions_used = ions_sat
            if DEBUG_DIAG:
                print(
                    "[WISE] MaxSAT unavailable (timeout/error); "
                    "falling back to SAT model at P*.",
                    flush=True,
                )

        # -------------------------------
        # Decode layouts a[r,P_star] from model_used
        # -------------------------------
        model_set = {lit for lit in model_used if lit > 0}

        def lit_true(v: int) -> bool:
            return v in model_set

        layouts: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            for d in range(n):
                for c in range(m):
                    found = None
                    for ion in ions_used:
                        if lit_true(var_a_used(r, P_star, d, c, ion)):
                            found = ion
                            break
                    if found is None:
                        raise RuntimeError(
                            f"Could not reconstruct cell (round={r}, d={d}, c={c})"
                        )
                    nxt[d, c] = found
            layouts.append(nxt)
            cur = nxt

        # After RC2:
        schedule = decode_wise_schedule_from_model(
            model=model_used,
            vpool=vpool_used,
            n=n,
            m=m,
            R=R,
            P_bound=P_star,
        )
        # print(schedule)
        freeze_seed_next = {
            "n_core0": n,
            "m_core0": m,
            "P_freeze0": P_star,
            "prev_vpool": vpool_used,
            "prev_true_set": model_set,
        }

        return layouts, schedule, freeze_seed_next
    



    
    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: "QCCDWiseArch",
        arrangement: Mapping["Trap", Sequence["Ion"]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        ignoreSpectators: bool = False,
        sat_schedule: List[Dict[str, Any]] = None,   # NEW: decoded schedule from RC2
    ) -> Tuple[Mapping[int, float], float]:
        """
        Shapes:
            rows = wiseArch.n, cols = wiseArch.m * wiseArch.k
            Arrays hold ion IDs (ints).

        If `sat_schedule` is provided, we:
            - execute exactly the swaps indicated by that schedule, in order,
            - charge heating/time per pass as per hardware model,
            - and assert that the final layout equals `newAssignment`.

        If `sat_schedule` is None, we fall back to the original heuristic
        odd–even reconfiguration (Phase B/C/D).
        """
        heatingRates: Dict[int, float] = {}
        for _, ions in arrangement.items():
            for ion in ions:
                heatingRates[ion.idx] = 0.0
        timeElapsed = 0.0

        row_swap_time = (
            Move.MOVING_TIME
            + Merge.MERGING_TIME
            + CrystalRotation.ROTATION_TIME
            + Split.SPLITTING_TIME
            + Move.MOVING_TIME
        )
        row_swap_heating = (
            Move.MOVING_TIME * Move.HEATING_RATE
            + Merge.MERGING_TIME * Merge.HEATING_RATE
            + CrystalRotation.ROTATION_TIME * CrystalRotation.HEATING_RATE
            + Split.SPLITTING_TIME * Split.HEATING_RATE
            + Move.MOVING_TIME * Move.HEATING_RATE
        )
        col_swap_time = (2 * JunctionCrossing.CROSSING_TIME) + (
            4 * JunctionCrossing.CROSSING_TIME + Move.MOVING_TIME
        ) * 2
        col_swap_heating_rate = (
            6 * JunctionCrossing.CROSSING_TIME * JunctionCrossing.HEATING_RATE
            + Move.MOVING_TIME * Move.HEATING_RATE
        )

        n = wiseArch.n  # rows
        m = wiseArch.m * wiseArch.k  # full columns (matches CNF m if built that way)
        k = wiseArch.k  # column stride for junction batching

        A = np.array(oldAssignment, dtype=int)  # current layout
        T = np.array(newAssignment, dtype=int)  # target layout

        spectatorIons: List[int] = []
        for ions in arrangement.values():
            spectatorIons.extend([ion.idx for ion in ions if isinstance(ion, SpectatorIon)])

        # ==========================================================
        # Phase A: parallel split (this is arch-specific overhead)
        # ==========================================================
        timeElapsed += Split.SPLITTING_TIME
        for idx in heatingRates.keys():
            heatingRates[idx] += Split.HEATING_RATE * Split.SPLITTING_TIME

        # If we have a SAT schedule, use it directly and SKIP Phases B/C/D.
        if sat_schedule is not None:
            acc_passes = 0

            for pass_idx, info in enumerate(sat_schedule):
                phase = info.get("phase", "H")
                h_swaps = info.get("h_swaps", [])
                v_swaps = info.get("v_swaps", [])

                # NOTE: to stay consistent with the SAT model, we SHOULD NOT
                #       skip swaps just because both ions are spectators.
                #       If you want spectators to be immobile, that must be
                #       encoded in the SAT itself.
                did_any_swap = False

                if phase == "H":
                    # All horizontal swaps in this pass are parallel
                    for (r, c) in h_swaps:
                        a = int(A[r, c])
                        b = int(A[r, c + 1])
                        # Optionally respect ignoreSpectators here, but that
                        # can deviate from the SAT layout. Safer to ignore it
                        # when using a SAT schedule:
                        if ignoreSpectators and (a in spectatorIons and b in spectatorIons):
                            continue
                        # Perform swap
                        A[r, c], A[r, c + 1] = b, a
                        heatingRates[a] += row_swap_heating
                        heatingRates[b] += row_swap_heating
                        did_any_swap = True

                    if did_any_swap:
                        timeElapsed += row_swap_time
                        acc_passes += 1

                elif phase == "V":
                    # All vertical swaps in this pass are parallel
                    for (r, c) in v_swaps:
                        a = int(A[r, c])
                        b = int(A[r + 1, c])
                        if ignoreSpectators and (a in spectatorIons and b in spectatorIons):
                            continue
                        A[r, c], A[r + 1, c] = b, a
                        heatingRates[a] += col_swap_heating_rate
                        heatingRates[b] += col_swap_heating_rate
                        did_any_swap = True

                    if did_any_swap:
                        timeElapsed += col_swap_time
                        acc_passes += 1

                else:
                    # Should never happen; phase is either H or V
                    pass

            # After executing the SAT schedule, check we reached the target.
            if not np.array_equal(A, T):
                print("[WARN] SAT-driven reconfig: final layout does NOT match newAssignment!")
                print("  A (final):")
                print(A)
                print("  T (target):")
                print(T)
                # You can raise if you want:
                # raise RuntimeError("SAT schedule did not realise target layout")

            print(
                f"RECONFIGURATION (SAT schedule): {acc_passes} passes were needed for the current reconfiguration round, "
                f"taking {timeElapsed} time and {heatingRates} heating"
            )
            return heatingRates, timeElapsed

        # ======================================================================
        # FALLBACK: original heuristic odd–even reconfiguration (unchanged)
        # ======================================================================

        # ---------- helper: odd-even passes ----------
        def row_pass_by_rank(even_phase: bool, row_rank: List[Dict[int, int]]) -> bool:
            maxSwapsInRow = 0
            start = 0 if even_phase else 1
            for r in range(n):
                swapsInRow = 0
                rank = row_rank[r]
                for c in range(start, m - 1, 2):
                    a = int(A[r, c])
                    b = int(A[r, c + 1])
                    if not ignoreSpectators and a in spectatorIons and b in spectatorIons:
                        continue
                    if rank[a] > rank[b]:
                        A[r, c], A[r, c + 1] = b, a
                        heatingRates[a] += row_swap_heating
                        heatingRates[b] += row_swap_heating
                        swapsInRow += 1
                if swapsInRow > maxSwapsInRow:
                    maxSwapsInRow = swapsInRow
            return maxSwapsInRow > 0

        def col_bucket_pass(
            even_phase: bool, bucket_mod: int, ion_to_dest_row: Dict[int, int]
        ) -> bool:
            maxSwapsInCol = 0
            start = 0 if even_phase else 1
            for c in range(bucket_mod, m, k):
                swapsInCol = 0
                for r in range(start, n - 1, 2):
                    a = int(A[r, c])
                    b = int(A[r + 1, c])
                    if not ignoreSpectators and a in spectatorIons and b in spectatorIons:
                        continue
                    if ion_to_dest_row[a] > ion_to_dest_row[b]:
                        A[r, c], A[r + 1, c] = b, a
                        heatingRates[a] += col_swap_heating_rate
                        heatingRates[b] += col_swap_heating_rate
                        swapsInCol += 1
                if swapsInCol > maxSwapsInCol:
                    maxSwapsInCol = swapsInCol
            return maxSwapsInCol > 0

        # ---------- destination row/col maps (ion -> dest row/col) ----------
        ion_to_dest_row: Dict[int, int] = {}
        ion_to_dest_col: Dict[int, int] = {}
        for r in range(n):
            for c in range(m):
                ion_to_dest_row[int(T[r, c])] = r
                ion_to_dest_col[int(T[r, c])] = c

        # ==========================================================
        # Phase B: ensure each column has unique destination rows
        # via m perfect matchings; then 1D odd-even per row.
        # ==========================================================
        desired_row_order = np.zeros_like(A)
        for r in range(n):
            for ionidx in A[r]:
                desired_row_order[r][ion_to_dest_col[ionidx]] = ionidx

        dest_rows = [[None] * m for _ in range(n)]
        for r in range(n):
            for c in range(m):
                dest_rows[r][c] = ion_to_dest_row[desired_row_order[r][c]]

        for c in range(m):
            assert len(set(dest_rows[r][c] for r in range(n))) == n

        for r in range(n):
            assert len(set(A[r]).difference(set(desired_row_order[r]))) == 0

        # Row ranks for Phase B permutation
        row_rank_phaseB: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_phaseB.append({ion: idx for idx, ion in enumerate(desired_row_order[r])})

        acc_cost = 0

        # Execute ≤ m odd–even steps to realise the permutation per row
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_phaseB)
            evenpass = row_pass_by_rank(False, row_rank_phaseB)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        # ==========================================================
        # Phase C: vertical odd–even with k-way parallel buckets.
        # ==========================================================
        for t in range(k):
            for _ in range(n):
                oddpass = col_bucket_pass(True, t, ion_to_dest_row)
                evenpass = col_bucket_pass(False, t, ion_to_dest_row)
                timeElapsed += oddpass * col_swap_time
                timeElapsed += evenpass * col_swap_time
                acc_cost += int(oddpass) + int(evenpass)
            if t < k - 1:
                # "Parrellel row reconfig"
                timeElapsed += k * row_swap_time
                for idx in heatingRates.keys():
                    heatingRates[idx] += row_swap_heating

        # ==========================================================
        # Phase D: final row-wise odd–even to exact target order.
        # ==========================================================
        row_rank_final: List[Dict[int, int]] = []
        for r in range(n):
            row_rank_final.append({ion: idx for idx, ion in enumerate(T[r, :])})
        for _ in range(m):
            oddpass = row_pass_by_rank(True, row_rank_final)
            evenpass = row_pass_by_rank(False, row_rank_final)
            timeElapsed += oddpass * row_swap_time
            timeElapsed += evenpass * row_swap_time
            acc_cost += int(oddpass) + int(evenpass)

        print(
            f"RECONFIGURATION: {acc_cost} passes were needed for the current reconfiguration round, "
            f"taking {timeElapsed} time and {heatingRates} heating"
        )
        return heatingRates, timeElapsed
    @classmethod
    def _runOddEvenReconfig2(
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
    
