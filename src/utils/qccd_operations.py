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
    Iterable,
    Union
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

# ---------- UNSAT core helpers ----------

class NoFeasibleLayoutError(RuntimeError):
    ...
class CoreGroups:
    """
    Manage assumption selectors for clause families to enable UNSAT core debug.
    Clauses guarded under selector s_g are added as (clause ∨ s_g).
    Solving with assumption ¬s_g activates the group; cores then map back to names.
    """

    def __init__(self, vpool: IDPool, enabled: bool, granularity: str = "coarse"):
        self.vpool = vpool
        self.enabled = enabled
        self.granularity = granularity
        self.sel: Dict[str, int] = {}
        self.meta: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def lit(self, name: str) -> int:
        if name not in self.sel:
            self.sel[name] = self.vpool.id(("grp", name))
        return self.sel[name]

    def add(
        self,
        formula: CNF,
        cl: List[int],
        group: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            formula.append(cl)
            return
        formula.append(cl + [self.lit(group)])
        if meta is not None:
            self.meta[group].append(meta)

def _sat_worker_from_file(
    cnf_path: str,
    result_path: str,
    assumptions: Optional[List[int]] = None,
):
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
            if assumptions is None:
                sat_ok = sat.solve()
            else:
                sat_ok = sat.solve(assumptions=assumptions)
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
    assumptions: Optional[List[int]] = None,
):
    """
    Run Minisat22 on 'cnf' in a separate process with a wall-clock timeout.

    Returns: (sat_ok, model, status)
      - status == "ok"        : sat_ok is True/False, model is list[int] or None.
      - status == "timeout"   : sat_ok, model are None (worker killed by timeout).
      - status == "error"     : sat_ok, model are None (worker crashed).
      - status == "user_abort": sat_ok, model are None (parent got KeyboardInterrupt
                                while waiting; worker terminated & cleaned up).

    If `assumptions` is provided, the SAT call uses those literals (usually the
    negated selector vars from CoreGroups) so guarded clauses remain active.
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

        p = mp.Process(
            target=_sat_worker_from_file,
            args=(cnf_path, result_path, assumptions),
        )
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
        cls, arrangement: Mapping[Trap, Sequence[Ion]], wiseArch: QCCDWiseArch, oldAssignment: Sequence[Sequence[int]], newAssignment: Sequence[Sequence[int]], schedule: List[Dict[str, Any]], initial_placement: bool = False
    ):
        heatingRates, reconfigTime = cls._runOddEvenReconfig(wiseArch, arrangement, oldAssignment, newAssignment, sat_schedule=schedule, initial_placement=initial_placement)
        reconfigTime=1e-20 if initial_placement else reconfigTime
        def run():
            for trap in arrangement.keys():
                while trap.ions:
                    trap.removeIon(trap.ions[0])
            for trap, ions in arrangement.items():
                for i, ion in enumerate(ions):
                    trap.addIon(ion, offset=i)
                    if initial_placement: 
                        continue
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
        max_sat_time = 1200.0,  
        active_ions: Set[int] = None,
        full_P_arr: List[List[Tuple[int, int]]]=[],
        ignore_initial_reconfig: bool = False,
        base_pmax_in: int = None,
        prev_pmax: int = None,
        grid_origin: Tuple[int, int] = (0, 0),
        boundary_adjacent: Optional[Dict[str, bool]] = None,
        cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
    ) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], int]:
        DEBUG_DIAG = True
        DEBUG_DIAG_DETAILED = False

        A_in = np.asarray(A_in, dtype=int)
        n, m = A_in.shape
        R = len(P_arr)
        optimize_round_start = 1 if (ignore_initial_reconfig and R > 0) else 0

        if BT is None:
            BT = [{} for _ in range(R)]
        if len(full_P_arr)==0:
            full_P_arr=P_arr

        if boundary_adjacent is None:
            boundary_adjacent = {"top": False, "bottom": False, "left": False, "right": False}
        else:
            boundary_adjacent = {
                "top": bool(boundary_adjacent.get("top", False)),
                "bottom": bool(boundary_adjacent.get("bottom", False)),
                "left": bool(boundary_adjacent.get("left", False)),
                "right": bool(boundary_adjacent.get("right", False)),
            }

        if cross_boundary_prefs is None or len(cross_boundary_prefs) != R:
            cross_boundary_prefs = [dict() for _ in range(R)]

        if base_pmax_in is None:
            base_pmax_in = R

        if prev_pmax is None:
            prev_pmax = 0

        row_offset = 0
        col_offset = 0
        if grid_origin is not None:
            row_offset, col_offset = grid_origin

        CAPACITY = k

        # -------------------------------
        # Basic sets of ions
        # -------------------------------
        row_of = {int(A_in[r, c]): r for r in range(n) for c in range(m)}
        col_of = {int(A_in[r, c]): c for r in range(n) for c in range(m)}
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
        first_block_idx = col_offset // CAPACITY
        last_block_idx = (col_offset + m - 1) // CAPACITY
        num_blocks = last_block_idx - first_block_idx + 1
    
        block_cells: List[List[Tuple[int, int]]] = []
        block_fully_inside: List[bool] = []
        block_widths: List[int] = []
        global_patch_start = col_offset
        global_patch_end = col_offset + m

        for b_local in range(num_blocks):
            b_global = first_block_idx + b_local
            global_start = b_global * CAPACITY
            global_end = (b_global + 1) * CAPACITY
            local_start = max(0, global_start - col_offset)
            local_end = min(m, global_end - col_offset)
            cells: List[Tuple[int, int]] = []
            for d in range(n):
                for j_local in range(local_start, local_end):
                    cells.append((d, j_local))
            block_cells.append(cells)
            block_widths.append(local_end-local_start)
            block_fully_inside.append(
                (global_start >= global_patch_start)
                and (global_end <= global_patch_end)
            )
        ions_set = set(ions)

        def compute_outer_pairs() -> Optional[List[List[Tuple[int, int]]]]:
            if not full_P_arr:
                return None
            outer: List[List[Tuple[int, int]]] = []
            for r in range(R):
                inner_set = set(P_arr[r]) if r < len(P_arr) else set()
                round_outer: List[Tuple[int, int]] = []
                full_round = full_P_arr[r] if r < len(full_P_arr) else []
                for pair in full_round:
                    if pair in inner_set:
                        continue
                    i1, i2 = pair
                    if i1 in ions_set or i2 in ions_set:
                        round_outer.append(pair)
                outer.append(round_outer)
            return outer

        outer_pairs = compute_outer_pairs()

        # -------------------------------
        # Structural CNF / WCNF builder for given P_bound
        # -------------------------------
        def _build_structural_cnf(
            P_bound: int,
            sum_bound_B: Optional[int] = None,
            use_wcnf: bool = False,
            add_boundary_soft: bool = False,
            phase_label: str = "",
            debug_skip_pair_constraints: bool = False,
            debug_allow_phase_flips: bool = False,
            optimize_round_start: int = 0,
            debug_core: bool = False,
            core_granularity: str = "coarse",
            debug_skip_cardinality: bool = False,
            debug_disable_pairs_rounds: Optional[Set[int]] = None,
            boundary_adjacent: Optional[Dict[str, bool]] = None,
            cross_boundary_prefs: Optional[List[Dict[int, Set[str]]]] = None,
            boundary_capacity_factor: float = 1.0,
        ):
            vpool = IDPool()

            # ------------- variable helpers -------------

            def var_a(r, p, krow, jcol, ion):
                return vpool.id(("a", r, p, krow, jcol, ion))

            def var_s_h(r, p, krow, jcol):
                return vpool.id(("s_h", r, p, krow, jcol))

            def var_s_v(r, p, krow, jcol):
                return vpool.id(("s_v", r, p, krow, jcol))

            def var_phase(r, p):
                return vpool.id(("phase", r, p))

            def var_row_end(r, ion, d):
                return vpool.id(("row_end", r, ion, d))

            def var_w_end(r, ion, b):
                return vpool.id(("w_end", r, ion, b))

            def var_u(r, p):
                return vpool.id(("u", r, p))

            def is_reserved(r, ion):
                return ion in BT[r]

            def ion_in_full_P_arr(r, ion):
                return any((ion in g) for g in full_P_arr[r])

            def ion_in_minor_P_arr(r, ion):
                return any((ion in g) for g in P_arr[r]) or is_reserved(r, ion)

            # ------------- choose CNF or WCNF -------------

            if use_wcnf:
                formula = WCNF()
            else:
                formula = CNF()

            grp = CoreGroups(
                vpool=vpool,
                enabled=(debug_core and not use_wcnf),
                granularity=core_granularity,
            )

            def add_hard(
                cl: List[int],
                group_name: str,
                meta: Optional[Dict[str, Any]] = None,
            ) -> None:
                if use_wcnf:
                    formula.append(cl)
                else:
                    grp.add(formula, cl, group_name, meta=meta)

            def add_soft(cl: List[int], weight: int = 1) -> None:
                if use_wcnf:
                    formula.append(cl, weight=weight)
                else:
                    raise RuntimeError("Soft clauses not allowed in pure CNF mode")

            if debug_disable_pairs_rounds is None:
                disable_pairs_rounds: Set[int] = set()
            else:
                disable_pairs_rounds = set(debug_disable_pairs_rounds)

            P_bounds = ([P_bound + n + m] * optimize_round_start + [P_bound] * (R - optimize_round_start))

            if boundary_adjacent is None:
                boundary_adjacent = {
                    "top": True,
                    "bottom": True,
                    "left": True,
                    "right": True,
                }
            else:
                boundary_adjacent = {
                    "top": bool(boundary_adjacent.get("top", False)),
                    "bottom": bool(boundary_adjacent.get("bottom", False)),
                    "left": bool(boundary_adjacent.get("left", False)),
                    "right": bool(boundary_adjacent.get("right", False)),
                }

            if cross_boundary_prefs is None:
                cross_boundary_prefs_norm: List[Dict[int, Set[str]]] = [dict() for _ in range(R)]
            else:
                cross_boundary_prefs_norm = []
                for r in range(R):
                    prefs_r = cross_boundary_prefs[r] if r < len(cross_boundary_prefs) else {}
                    normalized: Dict[int, Set[str]] = {}
                    for ion, dirs in prefs_r.items():
                        normalized[ion] = set(dirs)
                    cross_boundary_prefs_norm.append(normalized)

            half_h = max(1, n // 2)
            half_w = max(1, m // 2)

            def cells_for_direction(direction: str) -> List[Tuple[int, int]]:
                cells: List[Tuple[int, int]] = []
                if direction == "left":
                    if not boundary_adjacent["left"]:
                        return cells
                    for d in range(n):
                        for c in range(half_w):
                            cells.append((d, c))
                elif direction == "right":
                    if not boundary_adjacent["right"]:
                        return cells
                    start = max(0, m - half_w)
                    for d in range(n):
                        for c in range(start, m):
                            cells.append((d, c))
                elif direction == "top":
                    if not boundary_adjacent["top"]:
                        return cells
                    for d in range(half_h):
                        for c in range(m):
                            cells.append((d, c))
                elif direction == "bottom":
                    if not boundary_adjacent["bottom"]:
                        return cells
                    start = max(0, n - half_h)
                    for d in range(start, n):
                        for c in range(m):
                            cells.append((d, c))
                return cells

            # ------------------------------------------------------------------
            # (0) Global permutation: exactly one ion per cell AND each ion in exactly one cell
            # ------------------------------------------------------------------
            if not debug_skip_cardinality:
                # (0a) Exactly one ion per cell (r,p,k,j)
                for r in range(R):
                    g_cell = f"CARD_CELL:r{r}"
                    for p in range(P_bounds[r] + 1):
                        for krow in range(n):
                            for jcol in range(m):
                                lits = [var_a(r, p, krow, jcol, ion) for ion in ions]
                                enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                                for cl in enc.clauses:
                                    add_hard(cl, g_cell)

                g_cell_final = "CARD_CELL:FINAL"
                for krow in range(n):
                    for jcol in range(m):
                        lits = [var_a(R, 0, krow, jcol, ion) for ion in ions]
                        enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                        for cl in enc.clauses:
                            add_hard(cl, g_cell_final)

                # (0b) Each ion occupies exactly one cell (k,j) at every (r,p)
                for r in range(R):
                    g_ion_r = f"CARD_ION:r{r}"
                    for p in range(P_bounds[r] + 1):
                        for ion in ions:
                            lits = [var_a(r, p, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
                            enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                            for cl in enc.clauses:
                                add_hard(cl, g_ion_r)

                g_ion_final = "CARD_ION:FINAL"
                for ion in ions:
                    lits = [var_a(R, 0, krow, jcol, ion) for krow in range(n) for jcol in range(m)]
                    enc = CardEnc.equals(lits=lits, encoding=EncType.ladder, vpool=vpool)
                    for cl in enc.clauses:
                        add_hard(cl, g_ion_final)

            # ------------------------------------------------------------------
            # (1) Initial layout and inter-round chaining
            # ------------------------------------------------------------------

            # (1a) Initial layout: a[0,0] equals A_in
            for krow in range(n):
                for jcol in range(m):
                    ion0 = int(A_in[krow, jcol])
                    # Ion ion0 must be there
                    add_hard([var_a(0, 0, krow, jcol, ion0)], "INIT")
                    # No other ion may be there at (0,0)
                    for ion in ions:
                        if ion != ion0:
                            add_hard([-var_a(0, 0, krow, jcol, ion)], "INIT")

            # (1b) Chaining: a[r+1,0] <-> a[r,P_bounds[r]] for r=0..R-1
            for r in range(R):
                g_chain_r = f"CHAIN:r{r}"
                for krow in range(n):
                    for jcol in range(m):
                        for ion in ions:
                            a_next = var_a(r + 1, 0, krow, jcol, ion)
                            a_end  = var_a(r, P_bounds[r], krow, jcol, ion)
                            add_hard([-a_next, a_end], g_chain_r)
                            add_hard([-a_end, a_next], g_chain_r)

            # ------------------------------------------------------------------
            # (2) BT pinning at end-of-round (FINAL STATE ONLY)
            # ------------------------------------------------------------------

            for r in range(R):
                for ion in ions:
                    if is_reserved(r, ion):
                        d_fix, c_fix = BT[r][ion]
                        # Ion must occupy (d_fix,c_fix) at end-of-round r
                        add_hard([var_a(r, P_bounds[r], d_fix, c_fix, ion)], "BT")
                        # Ion cannot be anywhere else at end-of-round r
                        for krow in range(n):
                            for jcol in range(m):
                                if (krow, jcol) != (d_fix, c_fix):
                                    add_hard([-var_a(r, P_bounds[r], krow, jcol, ion)], "BT")

            # ------------------------------------------------------------------
            # (3) Phase structure: horizontal -> vertical
            # ------------------------------------------------------------------

            # (3a) Monotonicity: phase[r,p] -> phase[r,p+1] for each round r
            if not debug_allow_phase_flips:
                for r in range(R):
                    if P_bounds[r] > 1:
                        for p in range(P_bounds[r] - 1):
                            add_hard([-var_phase(r, p), var_phase(r, p + 1)], "PHASE_MONO")

            # ------------------------------------------------------------------
            # (4) Horizontal and (5) Vertical comparators + semantics + copy constraints
            # ------------------------------------------------------------------

            for r in range(R):
                for p in range(P_bounds[r]):
                    phase_p = var_phase(r,p)

                    # ---------------- Horizontal comparators (row-wise) ----------------
                    for krow in range(n):
                        for jcol in range(m - 1):
                            sh = var_s_h(r, p, krow, jcol)

                            # Gating: if phase[p] = 1 (vertical), horizontal comparators must be off
                            # phase[p] = 1 ⇒ ¬s_h  ==  (¬phase[p] ∨ ¬s_h)
                            add_hard([-phase_p, -sh], "H_GATE")

                            # Parity (odd-even) for horizontal comparisons:
                            # here we just use p%2 as horizontal parity:
                            #   if jcol % 2 != p % 2 then s_h must be 0
                            if jcol % 2 != (p % 2):
                                add_hard([-sh], "H_GATE")

                            # Forward-only SWAP semantics between (krow,jcol) and (krow,jcol+1)
                            # NOTE: we ONLY constrain the s=1 (swap) case here.
                            #       The s=0 (identity) case is handled by the horizontal copy
                            #       constraints below, which are guarded on ¬phase and ¬s_left/¬s_right.
                            for ion in ions:
                                a_cur_j = var_a(r, p, krow, jcol, ion)
                                a_cur_j1 = var_a(r, p, krow, jcol + 1, ion)
                                a_next_j = var_a(r, p + 1, krow, jcol, ion)
                                a_next_j1 = var_a(r, p + 1, krow, jcol + 1, ion)

                                # (H3) If s=1 and ion is in right cell at p, it must be in left at p+1:
                                #      (s ∧ a_cur_j1) -> a_next_j
                                add_hard([-sh, -a_cur_j1, a_next_j], "H_SEM")

                                # (H4) If s=1 and ion is in left cell at p, it must be in right at p+1:
                                #      (s ∧ a_cur_j) -> a_next_j1
                                add_hard([-sh, -a_cur_j, a_next_j1], "H_SEM")

                    # ---------------- Vertical comparators (column-wise) ----------------
                    for krow in range(n - 1):
                        for jcol in range(m):
                            sv = var_s_v(r, p, krow, jcol)

                            # Gating: if phase[p] = 0 (horizontal), vertical comparators must be off
                            # phase[p] = 0 ⇒ ¬s_v  ==  (phase[p] ∨ ¬s_v)
                            add_hard([phase_p, -sv], "V_GATE")

                            # Parity (odd-even) for vertical comparisons, using p%2 as v-index:
                            #   if krow % 2 != p % 2 then s_v must be 0
                            if krow % 2 != (p % 2):
                                add_hard([-sv], "V_GATE")

                            # Forward-only SWAP semantics between (krow,jcol) and (krow+1,jcol)
                            # Again, ONLY the s=1 case is constrained here; s=0 is enforced by
                            # the vertical copy constraints (phase ∧ ¬s_up ∧ ¬s_down).
                            for ion in ions:
                                a_cur_top = var_a(r, p, krow, jcol, ion)
                                a_cur_bot = var_a(r, p, krow + 1, jcol, ion)
                                a_next_top = var_a(r, p + 1, krow, jcol, ion)
                                a_next_bot = var_a(r, p + 1, krow + 1, jcol, ion)

                                # (V3) If s=1 and ion is in bottom cell at p, it must be in top at p+1:
                                #      (s ∧ a_cur_bot) -> a_next_top
                                add_hard([-sv, -a_cur_bot, a_next_top], "V_SEM")

                                # (V4) If s=1 and ion is in top cell at p, it must be in bottom at p+1:
                                #      (s ∧ a_cur_top) -> a_next_bot
                                add_hard([-sv, -a_cur_top, a_next_bot], "V_SEM")

            # -------- Copy constraints for non-participating cells (horizontal phase) --------
            for r in range(R):
                for p in range(P_bounds[r]):
                    phase_p = var_phase(r,p)
                    for krow in range(n):
                        for jcol in range(m):
                            # horizontal comparators that touch (krow,jcol)
                            s_left = var_s_h(r, p, krow, jcol - 1) if jcol > 0 else None
                            s_right = var_s_h(r, p, krow, jcol) if jcol < m - 1 else None

                            # antecedent: (¬phase[p] ∧ ¬s_left ∧ ¬s_right)
                            lits_ante_neg = [phase_p]  # this is ¬(¬phase), used on the clause side

                            if s_left is not None:
                                lits_ante_neg.append(s_left)
                            if s_right is not None:
                                lits_ante_neg.append(s_right)

                            for ion in ions:
                                a_cur = var_a(r, p, krow, jcol, ion)
                                a_next = var_a(r, p + 1, krow, jcol, ion)

                                # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_cur) -> a_next
                                # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_cur ∨ a_next)
                                add_hard(lits_ante_neg + [-a_cur, a_next], "H_COPY")

                                # (¬phase ∧ ¬s_left ∧ ¬s_right ∧ a_next) -> a_cur
                                # CNF: (phase ∨ s_left ∨ s_right ∨ ¬a_next ∨ a_cur)
                                add_hard(lits_ante_neg + [-a_next, a_cur], "H_COPY")

            # -------- Copy constraints for non-participating cells (vertical phase) --------
            for r in range(R):
                for p in range(P_bounds[r]):
                    phase_p = var_phase(r,p)
                    for krow in range(n):
                        for jcol in range(m):
                            # vertical comparators that touch (krow,jcol)
                            s_up = var_s_v(r, p, krow - 1, jcol) if krow > 0 else None
                            s_down = var_s_v(r, p, krow, jcol) if krow < n - 1 else None

                            # antecedent: (phase[p] ∧ ¬s_up ∧ ¬s_down)
                            lits_ante_neg = [-phase_p]  # this is ¬phase on the clause side

                            if s_up is not None:
                                lits_ante_neg.append(s_up)
                            if s_down is not None:
                                lits_ante_neg.append(s_down)

                            for ion in ions:
                                a_cur = var_a(r, p, krow, jcol, ion)
                                a_next = var_a(r, p + 1, krow, jcol, ion)

                                # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_cur) -> a_next
                                # CNF: (¬phase ∨ s_up ∨ s_down ∨ ¬a_cur ∨ a_next)
                                add_hard(lits_ante_neg + [-a_cur, a_next], "V_COPY")

                                # (phase ∧ ¬s_up ∧ ¬s_down ∧ a_next) -> a_cur
                                # CNF: (¬phase ∨ s_up ∨ s_down ∨ ¬a_next ∨ a_cur)
                                add_hard(lits_ante_neg + [-a_next, a_cur], "V_COPY")

            # ------------------------------------------------------------------
            # (5) End-of-round row/block abstraction and pair constraints
            # ------------------------------------------------------------------

            for r in range(R):
                g_rb_r = f"ROWBLOCK_LINK:r{r}"
                g_pair_r = f"PAIR_REQ:r{r}"

                # row_end / w_end linkage from final layout a[r,P_bounds[r]]
                for ion in ions:
                    # row_end
                    for d in range(n):
                        re = var_row_end(r, ion, d)
                        cell_lits = [var_a(r, P_bounds[r], d, j, ion) for j in range(m)]
                        add_hard([-re] + cell_lits, g_rb_r)
                        for aj in cell_lits:
                            add_hard([-aj, re], g_rb_r)

                    # w_end (global block alignment)
                    for b_local in range(num_blocks):
                        we = var_w_end(r, ion, b_local)
                        cell_list = block_cells[b_local]
                        cells = [var_a(r, P_bounds[r], d, j_local, ion) for (d, j_local) in cell_list]
                        if block_fully_inside[b_local] or (block_widths[b_local]>1):
                            add_hard([-we] + cells, g_rb_r)
                            for aj in cells:
                                add_hard([-aj, we], g_rb_r)
                        else:
                            add_hard([-we], g_rb_r)

                # Pairs must share same row and block (unless skipping for debug)
                if not debug_skip_pair_constraints:
                    if r in disable_pairs_rounds:
                        pass
                    else:
                        for (i1, i2) in P_arr[r]:
                            if i1 not in ions or i2 not in ions:
                                continue

                            # Same row
                            for d in range(n):
                                re1 = var_row_end(r, i1, d)
                                re2 = var_row_end(r, i2, d)
                                add_hard([-re1, re2], g_pair_r)
                                add_hard([-re2, re1], g_pair_r)
                            # Same block (global aligned); only enforce for fully covered blocks
                            for b_local in range(num_blocks):
                                we1 = var_w_end(r, i1, b_local)
                                we2 = var_w_end(r, i2, b_local)
                                add_hard([-we1, we2], g_pair_r)
                                add_hard([-we2, we1], g_pair_r)

            # ------------------------------------------------------------------
            # (6) Optional Level-3 soft clauses (boundary avoidance, swap-cost)
            # ------------------------------------------------------------------
            if use_wcnf and add_boundary_soft and (wB_row > 0 or wB_col > 0):
                inner_ions_per_round: List[Set[int]] = []
                for r in range(R):
                    inner_ions = set()
                    for (i1, i2) in P_arr[r]:
                        inner_ions.add(i1)
                        inner_ions.add(i2)
                    inner_ions.update(BT[r].keys())
                    inner_ions_per_round.append(inner_ions)

                for r in range(R):
                    inner_ions = inner_ions_per_round[r]
                    cross_prefs_r = cross_boundary_prefs[r] if r < len(cross_boundary_prefs) else {}
                    for ion in ions:
                        dirs = cross_prefs_r.get(ion)
                        if dirs:
                            for direction in dirs:
                                if direction in ("left", "right") and wB_col > 0:
                                    if not boundary_adjacent.get(direction, False):
                                        continue
                                    target_col = 0 if direction == "left" else m - 1
                                    lits = [var_a(r, P_bounds[r], d, target_col, ion) for d in range(n)]
                                    if lits:
                                        add_soft(lits, weight=wB_col)
                                if direction in ("top", "bottom") and wB_row > 0:
                                    if not boundary_adjacent.get(direction, False):
                                        continue
                                    target_row = 0 if direction == "top" else n - 1
                                    lits = [var_a(r, P_bounds[r], target_row, jcol, ion) for jcol in range(m)]
                                    if lits:
                                        add_soft(lits, weight=wB_row)

                        if ion in inner_ions:
                            if boundary_adjacent.get("left", False) and wB_col > 0:
                                for d in range(n):
                                    add_soft([-var_a(r, P_bounds[r], d, 0, ion)], weight=wB_col)
                            if boundary_adjacent.get("right", False) and wB_col > 0:
                                for d in range(n):
                                    add_soft([-var_a(r, P_bounds[r], d, m - 1, ion)], weight=wB_col)
                            if boundary_adjacent.get("top", False) and wB_row > 0:
                                for jcol in range(m):
                                    add_soft([-var_a(r, P_bounds[r], 0, jcol, ion)], weight=wB_row)
                            if boundary_adjacent.get("bottom", False) and wB_row > 0:
                                for jcol in range(m):
                                    add_soft([-var_a(r, P_bounds[r], n - 1, jcol, ion)], weight=wB_row)

            if cross_boundary_prefs_norm and any(boundary_adjacent.values()):
                factor = max(0.0, min(1.0, boundary_capacity_factor))
                dir_capacity: Dict[str, int] = {}
                if boundary_adjacent.get("top", False):
                    dir_capacity["top"] = int(round(half_h * m * factor))
                if boundary_adjacent.get("bottom", False):
                    dir_capacity["bottom"] = int(round(half_h * m * factor))
                if boundary_adjacent.get("left", False):
                    dir_capacity["left"] = int(round(half_w * n * factor))
                if boundary_adjacent.get("right", False):
                    dir_capacity["right"] = int(round(half_w * n * factor))

                ions_per_round_dir: Dict[Tuple[int, str], List[int]] = defaultdict(list)
                for r, prefs_r in enumerate(cross_boundary_prefs_norm):
                    for ion, dirs in prefs_r.items():
                        for direction in dirs:
                            if direction in dir_capacity:
                                ions_per_round_dir[(r, direction)].append(ion)

                for key in ions_per_round_dir:
                    ions_per_round_dir[key].sort()

                enforced_dirs_per_ion: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
                for (r, direction), ion_list in ions_per_round_dir.items():
                    cap = dir_capacity.get(direction, 0)
                    if cap <= 0:
                        continue
                    for ion in ion_list[:cap]:
                        enforced_dirs_per_ion[(r, ion)].add(direction)

                def _band_cells_for_dirs(directions: Set[str]) -> List[Tuple[int, int]]:
                    row_min, row_max = 0, n - 1
                    col_min, col_max = 0, m - 1
                    if "top" in directions:
                        row_max = min(row_max, half_h - 1)
                    if "bottom" in directions:
                        row_min = max(row_min, n - half_h)
                    if "left" in directions:
                        col_max = min(col_max, half_w - 1)
                    if "right" in directions:
                        col_min = max(col_min, m - half_w)
                    if row_min > row_max or col_min > col_max:
                        return []
                    return [
                        (rr, cc)
                        for rr in range(row_min, row_max + 1)
                        for cc in range(col_min, col_max + 1)
                    ]

                for r in range(R):
                    prefs_r = cross_boundary_prefs_norm[r]
                    if not prefs_r:
                        continue
                    P_final = P_bounds[r]
                    for ion in prefs_r.keys():
                        enforced_dirs = enforced_dirs_per_ion.get((r, ion))
                        if not enforced_dirs:
                            continue
                        cells = _band_cells_for_dirs(enforced_dirs)
                        if not cells:
                            union_cells: Set[Tuple[int, int]] = set()
                            for direction in enforced_dirs:
                                union_cells.update(_band_cells_for_dirs({direction}))
                            cells = list(union_cells)
                        if not cells:
                            if DEBUG_DIAG:
                                print(
                                    f"[CROSS_BOUNDARY] no valid cells for ion {ion} round {r} dirs={sorted(enforced_dirs)}; skipping",
                                    flush=True,
                                )
                            continue
                        clause = [var_a(r, P_final, d, c, ion) for (d, c) in cells]
                        add_hard(clause, "CROSS_BOUNDARY")

            # -------------------------------
            # Per-round pass usage helpers (u) and global Σ_r P_r bound
            # -------------------------------
            # Keep u[r,p] as-is: u <-> (OR of comparators)
            for r in range(R):
                for p in range(P_bounds[r]):
                    u_rp = var_u(r, p)

                    comp_lits: List[int] = []
                    for krow in range(n):
                        for jcol in range(m - 1):
                            comp_lits.append(var_s_h(r, p, krow, jcol))
                    for krow in range(n - 1):
                        for jcol in range(m):
                            comp_lits.append(var_s_v(r, p, krow, jcol))

                    if not comp_lits:
                        # No comparators exist at all in this pass: u must be false.
                        add_hard([-u_rp], "UTIL_U")
                        continue

                    # u[r,p] ↔ OR(comp_lits)
                    add_hard([-u_rp] + comp_lits, "UTIL_U")
                    for s_lit in comp_lits:
                        add_hard([-s_lit, u_rp], "UTIL_U")

            if sum_bound_B is not None and optimize_round_start < R:
                sum_u_lits: List[int] = []
                for r in range(optimize_round_start, R):
                    for p in range(P_bounds[r]):
                        sum_u_lits.append(var_u(r, p))

                total_slots = len(sum_u_lits)
                bound = min(sum_bound_B, total_slots)
                if bound < total_slots:
                    card_enc = CardEnc.atmost(
                        lits=sum_u_lits,
                        bound=bound,
                        encoding=EncType.totalizer,
                        vpool=vpool,
                    )
                    for clause in card_enc.clauses:
                        add_hard(clause, "SUM_BOUND")

            selectors = grp.sel if grp.enabled else {}
            group_meta = grp.meta if grp.enabled else {}
            return formula, vpool, ions, var_a, selectors, group_meta
     

        def decode_wise_schedule_from_model(
            model: List[int],
            vpool,
            n: int,
            m: int,
            R: int,
            P_bound: int,
        ) -> List[List[Dict[str, Any]]]:
            model_set = {lit for lit in model if lit > 0}
            P_bounds = ([P_bound+n+m]*int(ignore_initial_reconfig) + [P_bound]*(R-int(ignore_initial_reconfig)))

            def lit_true(v: int) -> bool:
                return v in model_set

            def var_s_h(r, p, krow, jcol):
                return vpool.id(("s_h", r, p, krow, jcol))

            def var_s_v(r, p, krow, jcol):
                return vpool.id(("s_v", r, p, krow, jcol))

            def var_phase(r, p):
                return vpool.id(("phase", r, p))

            schedule: List[List[Dict[str, Any]]] = [[] for _ in range(R)]
            for r in range(R):
                for p in range(P_bounds[r]):
                    phase_lit = var_phase(r, p)
                    is_vertical = (phase_lit <= vpool.top and lit_true(phase_lit))
                    phase = "V" if is_vertical else "H"

                    pass_info: Dict[str, Any] = {
                        "phase": phase,
                        "h_swaps": [],
                        "v_swaps": [],
                    }

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

                    schedule[r].append(pass_info)

            return schedule

        def extract_round_pass_usage(model, vpool, R, P_bound):
            model_set = {lit for lit in model if lit > 0}
            P_bounds = ([P_bound+n+m]*int(ignore_initial_reconfig) + [P_bound]*(R-int(ignore_initial_reconfig)))

            def lit_true(v: int) -> bool:
                return v in model_set

            def var_u(r, p):
                return vpool.id(("u", r, p))

            per_round: List[int] = []
            for r in range(R):
                count = 0
                for p in range(P_bounds[r]):
                    u_lit = var_u(r, p)
                    if u_lit <= vpool.top and lit_true(u_lit):
                        count += 1
                per_round.append(count)

            return per_round

        def wise_debug_boundary_stats(
            label: str,
            model: Iterable[int],
            vpool,
            var_a,
            ions: Iterable[int],
            n_sub: int,
            m_sub: int,
            R: int,
            P_bound: Union[int, Sequence[int]],
            inner_pairs: Optional[List[List[Tuple[int, int]]]] = None,
            outer_pairs: Optional[List[List[Tuple[int, int]]]] = None,
            boundary_adjacent: Optional[Dict[str, bool]] = None,
        ) -> Dict[str, Any]:
            """
            Patch-aware boundary stats. Counts how often ions involved in gates land
            on boundaries that actually have adjacent patches.
            """
            model_set = {l for l in model if l > 0}

            def lit_true(v: int) -> bool:
                return v in model_set

            if isinstance(P_bound, int):
                P_bounds = [P_bound] * R
            else:
                P_bounds = list(P_bound)
                if len(P_bounds) != R:
                    raise ValueError(
                        f"wise_debug_boundary_stats: len(P_bounds)={len(P_bounds)} != R={R}"
                    )

            last_row = n_sub - 1
            last_col = m_sub - 1

            if boundary_adjacent is None:
                boundary_adjacent = {
                    "top": True,
                    "bottom": True,
                    "left": True,
                    "right": True,
                }
            else:
                boundary_adjacent = {
                    "top": bool(boundary_adjacent.get("top", False)),
                    "bottom": bool(boundary_adjacent.get("bottom", False)),
                    "left": bool(boundary_adjacent.get("left", False)),
                    "right": bool(boundary_adjacent.get("right", False)),
                }

            total_positions = 0
            boundary_hits_row = 0
            boundary_hits_col = 0
            boundary_hits_corner = 0

            per_ion_counts: Dict[int, Dict[str, Any]] = {
                ion: {
                    "total": 0,
                    "row": 0,
                    "col": 0,
                    "corner": 0,
                    "inner_hits": 0,
                    "outer_hits": 0,
                }
                for ion in ions
            }

            ions_set = set(ions)

            for r in range(R):
                ions_r: Set[int] = set()
                if inner_pairs is not None and r < len(inner_pairs):
                    for (i1, i2) in inner_pairs[r]:
                        ions_r.add(i1)
                        ions_r.add(i2)
                if outer_pairs is not None and r < len(outer_pairs):
                    for (i1, i2) in outer_pairs[r]:
                        ions_r.add(i1)
                        ions_r.add(i2)
                if not ions_r:
                    ions_r = ions_set
                else:
                    ions_r &= ions_set

                if not ions_r:
                    continue

                inner_ions_r: Set[int] = set()
                outer_ions_r: Set[int] = set()
                if inner_pairs is not None and r < len(inner_pairs):
                    for (i1, i2) in inner_pairs[r]:
                        inner_ions_r.add(i1)
                        inner_ions_r.add(i2)
                if outer_pairs is not None and r < len(outer_pairs):
                    for (i1, i2) in outer_pairs[r]:
                        outer_ions_r.add(i1)
                        outer_ions_r.add(i2)

                p_final = P_bounds[r]

                for ion in ions_r:
                    if ion not in per_ion_counts:
                        per_ion_counts[ion] = {
                            "total": 0,
                            "row": 0,
                            "col": 0,
                            "corner": 0,
                            "inner_hits": 0,
                            "outer_hits": 0,
                        }
                    for d in range(n_sub):
                        for c in range(m_sub):
                            v = var_a(r, p_final, d, c, ion)
                            if not lit_true(v):
                                continue

                            total_positions += 1
                            per_ion_counts[ion]["total"] += 1

                            on_top = (d == 0 and boundary_adjacent["top"])
                            on_bottom = (d == last_row and boundary_adjacent["bottom"])
                            on_left = (c == 0 and boundary_adjacent["left"])
                            on_right = (c == last_col and boundary_adjacent["right"])

                            on_row_boundary = on_top or on_bottom
                            on_col_boundary = on_left or on_right

                            if on_row_boundary:
                                boundary_hits_row += 1
                                per_ion_counts[ion]["row"] += 1
                            if on_col_boundary:
                                boundary_hits_col += 1
                                per_ion_counts[ion]["col"] += 1
                            if on_row_boundary and on_col_boundary:
                                boundary_hits_corner += 1
                                per_ion_counts[ion]["corner"] += 1

                            if ion in inner_ions_r:
                                per_ion_counts[ion]["inner_hits"] += 1
                            if ion in outer_ions_r:
                                per_ion_counts[ion]["outer_hits"] += 1

            total_positions = max(total_positions, 1)

            summary = {
                "label": label,
                "n_sub": n_sub,
                "m_sub": m_sub,
                "R": R,
                "P_bounds": P_bounds,
                "total_positions": total_positions,
                "boundary_hits_row": boundary_hits_row,
                "boundary_hits_col": boundary_hits_col,
                "boundary_hits_corner": boundary_hits_corner,
                "frac_row": boundary_hits_row / total_positions,
                "frac_col": boundary_hits_col / total_positions,
                "frac_corner": boundary_hits_corner / total_positions,
                "boundary_adjacent": boundary_adjacent,
                "per_ion_counts": per_ion_counts,
            }

            print(
                f"[WISE-DEBUG] boundary-stats {label}: "
                f"subgrid={n_sub}x{m_sub}, "
                f"pos={total_positions}, "
                f"row_hits={boundary_hits_row} ({summary['frac_row']:.3f}), "
                f"col_hits={boundary_hits_col} ({summary['frac_col']:.3f}), "
                f"corner_hits={boundary_hits_corner} ({summary['frac_corner']:.3f}), "
                f"adjacent={boundary_adjacent}"
            )

            worst = sorted(
                per_ion_counts.items(),
                key=lambda kv: (kv[1]["row"] + kv[1]["col"]),
                reverse=True,
            )[:5]
            for ion, stats in worst:
                if stats["total"] == 0:
                    continue
                fr = stats["row"] / stats["total"]
                fc = stats["col"] / stats["total"]
                print(
                    f"[WISE-DEBUG]   ion {ion}: total={stats['total']}, "
                    f"row={stats['row']} ({fr:.3f}), "
                    f"col={stats['col']} ({fc:.3f}), "
                    f"corner={stats['corner']}, "
                    f"inner_hits={stats['inner_hits']}, "
                    f"outer_hits={stats['outer_hits']}"
                )

            return summary

        def _enumerate_pmax_configs(
            P_min: int,
            P_max_limit: int,
            step: int = 1,
            *,
            capacity_steps: int = 6,
            capacity_min: float = 0.0,
        ) -> Iterable[Tuple[int, float]]:
            """
            Enumerate (P_max, boundary_capacity_factor) pairs. The capacity factor
            scales the number of ions that are forced into boundary bands for
            CROSS_BOUNDARY constraints. The final factor is capacity_min (typically 0).
            """
            if capacity_steps <= 1:
                factors = [1.0]
            else:
                factors = [
                    max(
                        capacity_min,
                        1.0 - i * (1.0 - capacity_min) / (capacity_steps - 1),
                    )
                    for i in range(capacity_steps)
                ]

            for P_max in range(P_min, P_max_limit + 1, max(1, step)):
                for factor in factors:
                    yield (P_max, factor)

        chosen_solution = None
        base_pmax = max(max(base_pmax_in, 1), prev_pmax)
        limit_pmax = base_pmax + n + m
        configs = _enumerate_pmax_configs(
            base_pmax,
            limit_pmax,
            step=max(int(np.floor((limit_pmax-base_pmax)/4)),1),
            capacity_steps=6,
            capacity_min=0.0,
        )

        chosen_boundary_capacity_factor = 1.0

        for (P_max, boundary_capacity_factor) in configs:
            rounds_under_sum_local = max(1, R - optimize_round_start)
            B_lo = 0
            B_hi = rounds_under_sum_local * P_max
            sum_star = None

            if DEBUG_DIAG:
                print(
                    f"[WISE] starting binary search for ΣP in [{B_lo}, {B_hi}] "
                    f"(opt rounds start={optimize_round_start}) with P_max={P_max}, "
                    f"boundary_capacity_factor={boundary_capacity_factor:.2f}",
                    flush=True,
                )

            while B_lo <= B_hi:
                B_mid = (B_lo + B_hi) // 2

                (
                    cnf_mid,
                    vpool_mid,
                    ions_mid,
                    var_a_mid,
                    grp_sel_mid,
                    grp_meta_mid,
                ) = _build_structural_cnf(
                    P_max,
                    sum_bound_B=B_mid,
                    use_wcnf=False,
                    add_boundary_soft=False,
                    phase_label=f"ΣP={B_mid}/SAT",
                    optimize_round_start=optimize_round_start,
                    debug_core=True,
                    core_granularity="coarse",
                    debug_skip_cardinality=False,
                    boundary_adjacent=boundary_adjacent,
                    cross_boundary_prefs=cross_boundary_prefs,
                    boundary_capacity_factor=boundary_capacity_factor,
                )

                assumptions_mid = (
                    [-lit for lit in grp_sel_mid.values()] if grp_sel_mid else None
                )

                t_sat_start = time.time()
                sat_ok, model_mid, status_sat = run_sat_with_timeout_file(
                    cnf_mid,
                    timeout_s=max_sat_time,
                    debug_prefix=None,
                    assumptions=assumptions_mid,
                )

                # with Minisat22(bootstrap_with=cnf_mid.clauses) as sat:
                #     sat_ok = sat.solve(assumptions=assumptions_mid)
                #     model_mid = sat.get_model() if sat_ok else None
                #     status_sat = "ok" if sat_ok else "error"
                t_sat_end = time.time()

                if DEBUG_DIAG:
                    if hasattr(cnf_mid, "clauses"):
                        n_clauses = len(cnf_mid.clauses)
                    else:
                        n_clauses = len(cnf_mid.hard)  # just in case
                    print(
                        f"[WISE]  test ΣP={B_mid}, P_max={P_max}: status={status_sat}, SAT={sat_ok}, "
                        f"vars={vpool_mid.top}, clauses={n_clauses}, "
                        f"time={t_sat_end - t_sat_start:.3f}s",
                        flush=True,
                    )
                 
                if sat_ok:
                    chosen_solution = (
                        cnf_mid,
                        vpool_mid,
                        ions_mid,
                        var_a_mid,
                        model_mid,
                        B_mid,
                        P_max,
                        boundary_capacity_factor,
                    )
                    chosen_boundary_capacity_factor = boundary_capacity_factor
                    # Record this as the best so far and tighten upper bound
                    sum_star = B_mid
                    B_hi = B_mid - 1
                else:
                    # try:
                    #     with Minisat22(bootstrap_with=cnf_mid.clauses) as s:
                    #         assumptions = [-lit for lit in grp_sel_mid.values()]
                    #         ok = s.solve(assumptions=assumptions)
                    #         if not ok:
                    #             core = s.get_core() or []
                    #             inv = {lit: name for name, lit in grp_sel_mid.items()}

                    #             by_group: Dict[str, Set[int]] = {}
                    #             culprit_rounds: Set[int] = set()
                    #             culprit_fullnames: List[str] = []

                    #             for a in core:
                    #                 var = abs(a)
                    #                 fullname = inv.get(var)
                    #                 if fullname is None:
                    #                     continue
                    #                 culprit_fullnames.append(fullname)
                    #                 parts = fullname.split(":")
                    #                 base = parts[0]

                    #                 rounds_here: Set[int] = set()
                    #                 for part in parts[1:]:
                    #                     if part.startswith("r"):
                    #                         try:
                    #                             rounds_here.add(int(part[1:]))
                    #                         except ValueError:
                    #                             pass

                    #                 if rounds_here:
                    #                     by_group.setdefault(base, set()).update(rounds_here)
                    #                     culprit_rounds.update(rounds_here)
                    #                 else:
                    #                     by_group.setdefault(base, set())

                    #             # High-level summary
                    #             print(f"[UNSAT-CORE] ΣP={B_mid} → groups:", flush=True)
                    #             for base in sorted(by_group.keys()):
                    #                 rs = sorted(by_group[base])
                    #                 if rs:
                    #                     msg = f"  {base}: rounds {rs[0]}..{rs[-1]} (|R|={len(rs)})"
                    #                 else:
                    #                     msg = f"  {base}: no round tag"
                    #                 print(msg, flush=True)

                    #             if culprit_rounds:
                    #                 print(
                    #                     f"[UNSAT-CORE] culprit rounds (union): {sorted(culprit_rounds)}",
                    #                     flush=True,
                    #                 )

                    #             # Brief per-round context (only first few rounds)
                    #             for r_bad in sorted(culprit_rounds)[:5]:
                    #                 if 0 <= r_bad < len(P_arr):
                    #                     print(
                    #                         f"[UNSAT-CORE]   r={r_bad}, |P_arr[{r_bad}]| = {len(P_arr[r_bad])}",
                    #                         flush=True,
                    #                     )
                    #                 if 0 <= r_bad < len(BT):
                    #                     print(
                    #                         f"[UNSAT-CORE]   r={r_bad}, |BT[{r_bad}]|    = {len(BT[r_bad])}",
                    #                         flush=True,
                    #                     )

                    #             culprit_bases = set(by_group.keys())
                    #             hints = []
                    #             if "PAIR_REQ" in culprit_bases:
                    #                 hints.append("PAIR_REQ (check BT vs P_arr)")
                    #             if "CARD_CELL" in culprit_bases or "CARD_ION" in culprit_bases:
                    #                 hints.append("CARD_* (global permutation)")
                    #             if "PHASE_MONO" in culprit_bases:
                    #                 hints.append("PHASE_MONO (phase monotonicity)")
                    #             if "H_GATE" in culprit_bases or "V_GATE" in culprit_bases:
                    #                 hints.append("H_GATE/V_GATE (parity/gating)")
                    #             if "ROWBLOCK_LINK" in culprit_bases:
                    #                 hints.append("ROWBLOCK_LINK (row/block linkage vs BT/pairs)")
                    #             if "CROSS_BOUNDARY" in culprit_bases:
                    #                 hints.append("CROSS_BOUNDARY (cross-patch boundary bands / prefs)")

                    #             if hints:
                    #                 print("[UNSAT-CORE] key groups in core:", ", ".join(hints), flush=True)

                    #         else:
                    #             print(
                    #                 "[UNSAT-CORE] Unexpected: SAT under assumptions while previous solver said UNSAT.",
                    #                 flush=True,
                    #             )
                    # except Exception as e:
                    #     print(f"[UNSAT-CORE] core extraction error: {e}", flush=True)
                    # UNSAT: need more passes in aggregate
                    B_lo = B_mid + 1
            
            if sum_star is not None:
                break
        

        if sum_star is None:
            # try:
            #     with Minisat22(bootstrap_with=cnf_mid.clauses) as s:
            #         assumptions = [-lit for lit in grp_sel_mid.values()]
            #         ok = s.solve(assumptions=assumptions)
            #         if not ok:
            #             core = s.get_core() or []
            #             inv = {lit: name for name, lit in grp_sel_mid.items()}

            #             by_group: Dict[str, Set[int]] = {}
            #             culprit_rounds: Set[int] = set()
            #             culprit_fullnames: List[str] = []

            #             for a in core:
            #                 var = abs(a)
            #                 fullname = inv.get(var)
            #                 if fullname is None:
            #                     continue
            #                 culprit_fullnames.append(fullname)
            #                 parts = fullname.split(":")
            #                 base = parts[0]

            #                 rounds_here: Set[int] = set()
            #                 for part in parts[1:]:
            #                     if part.startswith("r"):
            #                         try:
            #                             rounds_here.add(int(part[1:]))
            #                         except ValueError:
            #                             pass

            #                 if rounds_here:
            #                     by_group.setdefault(base, set()).update(rounds_here)
            #                     culprit_rounds.update(rounds_here)
            #                 else:
            #                     by_group.setdefault(base, set())

            #             # High-level summary
            #             print(f"[UNSAT-CORE] ΣP={B_mid} → groups:", flush=True)
            #             for base in sorted(by_group.keys()):
            #                 rs = sorted(by_group[base])
            #                 if rs:
            #                     msg = f"  {base}: rounds {rs[0]}..{rs[-1]} (|R|={len(rs)})"
            #                 else:
            #                     msg = f"  {base}: no round tag"
            #                 print(msg, flush=True)

            #             if culprit_rounds:
            #                 print(
            #                     f"[UNSAT-CORE] culprit rounds (union): {sorted(culprit_rounds)}",
            #                     flush=True,
            #                 )

            #             # Brief per-round context (only first few rounds)
            #             for r_bad in sorted(culprit_rounds)[:5]:
            #                 if 0 <= r_bad < len(P_arr):
            #                     print(
            #                         f"[UNSAT-CORE]   r={r_bad}, |P_arr[{r_bad}]| = {len(P_arr[r_bad])}",
            #                         flush=True,
            #                     )
            #                 if 0 <= r_bad < len(BT):
            #                     print(
            #                         f"[UNSAT-CORE]   r={r_bad}, |BT[{r_bad}]|    = {len(BT[r_bad])}",
            #                         flush=True,
            #                     )

            #             culprit_bases = set(by_group.keys())
            #             hints = []
            #             if "PAIR_REQ" in culprit_bases:
            #                 hints.append("PAIR_REQ (check BT vs P_arr)")
            #             if "CARD_CELL" in culprit_bases or "CARD_ION" in culprit_bases:
            #                 hints.append("CARD_* (global permutation)")
            #             if "PHASE_MONO" in culprit_bases:
            #                 hints.append("PHASE_MONO (phase monotonicity)")
            #             if "H_GATE" in culprit_bases or "V_GATE" in culprit_bases:
            #                 hints.append("H_GATE/V_GATE (parity/gating)")
            #             if "ROWBLOCK_LINK" in culprit_bases:
            #                 hints.append("ROWBLOCK_LINK (row/block linkage vs BT/pairs)")
            #             if "CROSS_BOUNDARY" in culprit_bases:
            #                 hints.append("CROSS_BOUNDARY (cross-patch boundary bands / prefs)")

            #             if hints:
            #                 print("[UNSAT-CORE] key groups in core:", ", ".join(hints), flush=True)

            #         else:
            #             print(
            #                 "[UNSAT-CORE] Unexpected: SAT under assumptions while previous solver said UNSAT.",
            #                 flush=True,
            #             )
            # except Exception as e:
            #     print(f"[UNSAT-CORE] core extraction error: {e}", flush=True)
            raise NoFeasibleLayoutError("No feasible layout for any Σ_r P_r bound in [0, R * P_max].")

        (
            _,
            vpool_sat,
            ions_sat,
            var_a_sat,
            sat_model_star,
            best_sum_bound,
            P_max,
            chosen_boundary_capacity_factor,
        ) = chosen_solution
        if DEBUG_DIAG:
            print(f"[WISE] minimal ΣP found: {best_sum_bound} (P_max={P_max})", flush=True)

        pass_horizon = P_max
        P_bounds = ([pass_horizon + n + m] * optimize_round_start + [pass_horizon] * (R - optimize_round_start))

        layouts_before: List[np.ndarray] = []
        cur = A_in.copy()

        for r in range(R):
            nxt = np.empty_like(cur)
            for d in range(n):
                for c in range(m):
                    found = None
                    for ion in ions_sat:
                        if var_a_sat(r, pass_horizon, d, c, ion) in sat_model_star:
                            found = ion
                            break
                    if found is None:
                        raise RuntimeError(
                            f"Could not reconstruct cell (round={r}, d={d}, c={c})"
                        )
                    nxt[d, c] = found
            layouts_before.append(nxt)
            cur = nxt

        _ = wise_debug_boundary_stats(
            label="BEFORE MAXSAT",
            model=sat_model_star,
            vpool=vpool_sat,
            var_a=var_a_sat,
            ions=ions,
            n_sub=n,
            m_sub=m,
            R=R,
            P_bound=P_bounds,
            inner_pairs=P_arr,
            outer_pairs=outer_pairs,
            boundary_adjacent=boundary_adjacent,
        )
        # -------------------------------
        # Level 3: MaxSAT refinement under ΣP*
        # -------------------------------
        ENABLE_MAXSAT = False
        if ENABLE_MAXSAT:
            if DEBUG_DIAG:
                print(
                    f"[WISE] building WCNF at ΣP*={best_sum_bound}, P_max={pass_horizon} for MaxSAT...",
                    flush=True,
                )
            t_build_start = time.time()
            wcnf, vpool_w, ions_w, var_a_w, _, _ = _build_structural_cnf(
                pass_horizon,
                sum_bound_B=best_sum_bound,
                use_wcnf=True,
                add_boundary_soft=True,
                phase_label=f"ΣP*={best_sum_bound}/WCNF",
                optimize_round_start=optimize_round_start,
                debug_skip_cardinality=False,
                boundary_adjacent=boundary_adjacent,
                cross_boundary_prefs=cross_boundary_prefs,
                boundary_capacity_factor=chosen_boundary_capacity_factor,
            )
            t_build_end = time.time()

            if DEBUG_DIAG:
                print(
                    f"[WISE] WCNF built: vars={wcnf.nv}, hard={len(wcnf.hard)}, "
                    f"soft={len(wcnf.soft)}, time={t_build_end - t_build_start:.3f}s",
                    flush=True,
                )

            rc2 = RC2(wcnf)
            model_rc2 = rc2.compute()
            cost_rc2 = rc2.cost if model_rc2 is not None else None
            status_rc2 = "ok" if model_rc2 is not None else "error"

            if DEBUG_DIAG:
                print(
                    f"[WISE] RC2 status={status_rc2}, opt_cost={cost_rc2}",
                    flush=True,
                )

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
        else:
            model_used = sat_model_star
            vpool_used = vpool_sat
            var_a_used = var_a_sat
            ions_used = ions_sat
            if DEBUG_DIAG:
                print("[WISE] MaxSAT disabled; using SAT model at P*", flush=True)

        # Decide which ions are "core" for this slice.
        # A reasonable default: all active ions that lie entirely in the current subgrid.
        # If you already have a list `core_ions_this_slice`, reuse that here.
        core_ions_this_slice = ions   # or a filtered subset if you prefer

        _ = wise_debug_boundary_stats(
            label="AFTER MAXSAT",
            model=model_used,
            vpool=vpool_used,
            var_a=var_a_used,
            ions=ions_used,
            n_sub=n,
            m_sub=m,
            R=R,
            P_bound=P_bounds,
            inner_pairs=P_arr,
            outer_pairs=outer_pairs,
            boundary_adjacent=boundary_adjacent,
        )


        # -------------------------------
        # Decode layouts a[r,P_max] from model_used
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
                        if lit_true(var_a_used(r, P_bounds[r], d, c, ion)):
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
            P_bound=pass_horizon,
        )

        row_offset, col_offset = grid_origin
        if row_offset != 0 or col_offset != 0:
            for round_schedule in schedule:
                for pass_info in round_schedule:
                    if "h_swaps" in pass_info:
                        pass_info["h_swaps"] = [
                            (r + row_offset, c + col_offset) for (r, c) in pass_info["h_swaps"]
                        ]
                    if "v_swaps" in pass_info:
                        pass_info["v_swaps"] = [
                            (r + row_offset, c + col_offset) for (r, c) in pass_info["v_swaps"]
                        ]
        # print(schedule)
        per_round_z = extract_round_pass_usage(
            model_used,
            vpool_used,
            R,
            pass_horizon,
        )
        sum_all = sum(per_round_z)
        sum_tail = sum(per_round_z[optimize_round_start:])

        print(
            f"[WISE] ΣP per round: {per_round_z}, "
            f"Σ_all={sum_all}, Σ_tail={sum_tail}, "
            f"best_sum_bound={best_sum_bound}"
        )

        return layouts, schedule, pass_horizon
    



    
    @classmethod
    def _runOddEvenReconfig(
        cls,
        wiseArch: "QCCDWiseArch",
        arrangement: Mapping["Trap", Sequence["Ion"]],
        oldAssignment: Sequence[Sequence[int]],
        newAssignment: Sequence[Sequence[int]],
        ignoreSpectators: bool = False,
        sat_schedule: List[Dict[str, Any]] = None,   # NEW: decoded schedule from RC2
        initial_placement: bool = False
    ) -> Tuple[Mapping[int, float], float]:
        # Schedule-aware reconfiguration fallback when SAT results are available.
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
            if not initial_placement:
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

        if not initial_placement:
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
        # Basic deterministic reconfiguration without SAT solver; legacy helper.
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
    
