
import numpy as np
import numpy.typing as npt
from typing import (
    Sequence,
    List,
    Tuple,
    Optional,
    Mapping,
    Callable,
    Union,
    Any,
    Set,
    Dict,
)
from matplotlib import pyplot as plt
import networkx as nx
import enum
from matplotlib.patches import Ellipse
import abc
import stim
from scipy.spatial import distance
import pymatching
from qccd_nodes import *
from qccd_operations import *
from qccd_operations_on_qubits import *

class QCCDArch:
    # TODO need to abstract this to support different layouts of QCCD architectures
    SIZING = 1
    JUNCTION_SIZE = 800 * SIZING
    ION_SIZE = 800 * SIZING
    FONT_SIZE = 14 * SIZING
    WINDOW_SIZE = 30 * SIZING, 24 * SIZING
    TRAP_WIDTH = 15 * SIZING
    EDGE_WIDTH = SIZING

    HIGHLIGHT_COLOR = "yellow"
    HIGHLIGHT_NODE_SIZE = 4000 * SIZING
    JUNCTION_SHAPE = "s"
    ION_SHAPE = "o"
    DEFAULT_ALPHA = 0.5
    PADDING = 0.6

    N_ITERS = 5000

    def __init__(self):
        self._trapEdges: Mapping[int, Sequence[Tuple[int, int]]] = {}
        self._crossingEdges: Mapping[Tuple[int, int], Crossing] = {}
        self._crossings: List[Crossing] = []
        self._manipulationTraps: List[ManipulationTrap] = []
        self._junctions: List[Junction] = []
        self._nextIdx = 0
        self._routingTable: Mapping[int, Mapping[int, Sequence[Operation]]] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._inActiveEdges: List[int] = []
        self._centralities: Mapping
        self._originalArrangement: Dict[Ion, Trap] = {}

    @property
    def routingTable(self):
        return self._routingTable

    @property
    def ions(self) -> Mapping[int, Ion]:
        ions = {}
        for t in self._manipulationTraps:
            ions.update([(ion.idx, ion) for ion in t.ions])
        for j in self._junctions:
            ions.update([(ion.idx, ion) for ion in j.ions])
        for c in self._crossings:
            if c.ion:
                ions[c.ion.idx] = c.ion
        return ions

    @property
    def nodes(self) -> Mapping[int, QCCDNode]:
        cs = {}
        for t in self._manipulationTraps:
            cs[t.idx] = t
        for j in self._junctions:
            cs[j.idx] = j
        return cs

    def addEdge(self, source: QCCDNode, target: QCCDNode) -> Crossing:
        crossing = Crossing(self._nextIdx, source, target)
        self._crossings.append(crossing)
        self._nextIdx += 1
        return crossing

    def addManipulationTrap(
        self,
        x: int,
        y: int,
        ions: Sequence[Ion],
        color: str = ManipulationTrap.DEFAULT_COLOR,
        isHorizontal: bool = ManipulationTrap.DEFAULT_ORIENTATION,
        spacing: int = ManipulationTrap.DEFAULT_SPACING,
        capacity: int = ManipulationTrap.DEFAULT_CAPACITY,
    ) -> Trap:
        trap = ManipulationTrap(
            self._nextIdx,
            x,
            y,
            ions,
            color=color,
            isHorizontal=isHorizontal,
            spacing=spacing * self.SIZING,
            capacity=capacity,
        )
        for ion in trap.ions:
            self._originalArrangement[ion] = trap
        self._manipulationTraps.append(trap)
        self._nextIdx += len(ions) + 1
        return trap

    def addJunction(
        self,
        x: int,
        y: int,
        color: str = Junction.DEFAULT_COLOR,
        label: str = Junction.DEFAULT_LABEL,
        capacity: int = Junction.DEFAULT_CAPACITY,
    ) -> Junction:
        junction = Junction(
            self._nextIdx, x, y, color=color, label=label, capacity=capacity
        )
        self._junctions.append(junction)
        self._nextIdx += 1
        return junction


    def decideDestinationTrap(self, ion1: Ion, ion2: Ion) -> Trap:
        destIon = max(
            (ion1, ion2), key=lambda ion: ion.label[0]=='D'
        )
        return destIon.parent

    def decideDestinationTrapForRouting(self, ion1: Ion, ion2: Ion) -> Trap:
        destIon = max(
            (ion1, ion2), key=lambda ion: ion.parent.capacity - ion.parent.numIons 
        )
        return destIon.parent

    def route(self, ion: Ion, trap: ManipulationTrap) -> Sequence[Operation]:
        src = ion.idx
        dest = trap.idx
        if dest in self._routingTable[src]:
            return self._routingTable[src][dest]
        if not nx.has_path(self._graph, src, dest):
            self._routingTable[src][dest] = [False]
            return [False]
        paths = nx.all_shortest_paths(self._graph, src, dest)
        if self._centralities is None:
            self._centralities = nx.edge_betweenness_centrality(self._graph)
        best_path = sorted(
            paths,
            key=lambda path: sum(
                self._centralities[n1, n2] for n1, n2 in zip(path[:-1], path[1:])
            ),
        )[0]
        ops = []
        for n1, n2 in zip(best_path[:-1], best_path[1:]):
            op_ = self._graph.edges[n1, n2]["operations"]
            op_ = [
                (
                    GateSwap.physicalOperation(
                        trap=o._trap, ion1=self.ions[src],ion2= o._ion2
                    )
                    if isinstance(o, GateSwap)
                    else o
                )
                for o in op_
            ]
            ops.extend(op_)
        self._routingTable[src][dest] = ops
        return ops
        
    def pathFreeCapacity(self, path: Sequence[int]) -> int:
        freeCapacity = np.inf
        currentNode = (
            self.nodes[path[0]] if path[0] in self.nodes else self.ions[path[0]].parent
        )
        for x in path:
            node = self.nodes[x] if x in self.nodes else self.ions[x].parent
            if node != currentNode:
                # FIXME not using desired capacity since ions live in these traps temporarily
                freeCapacity = min(node.capacity - node.numIons, freeCapacity)
        return freeCapacity

    def forward(self, ion: Ion, trap: Trap) -> Sequence[Operation]:
        n1 = ion.idx
        n1Idx = ion.parent.idx
        if n1Idx == trap.idx:
            return []
        shortestPathStrings = [
            path for path in nx.all_shortest_paths(self._graph, ion.idx, trap.idx)
        ]
        bestScore = -np.inf
        bestOps = []
        bestFreeCapacity = 0
        outEdgesDict = dict(self._graph.out_edges._adjdict)
        hopSet_ = []
        opsSet_ = []
        for n2 in (ion.parent.nodes[1], ion.parent.nodes[-1]):
            opsStart_ = [] if n1 == n2 else list(outEdgesDict[n1][n2]["operations"])
            n2Idx = n2 if n2 in self.nodes else self.ions[n2].parent.idx
            hopStart_ = [n1, n2] if opsStart_ else [n1]
            if n1Idx != n2Idx:
                hopSet_.append(hopStart_)
                opsSet_.append(opsStart_)
            for n3, data2 in dict(outEdgesDict[n2]).items():
                n3Idx = n3 if n3 in self.nodes else self.ions[n3].parent.idx
                if n3Idx != n2Idx:
                    hopSet_.append(hopStart_ + [n3])
                    opsSet_.append(opsStart_ + list(data2["operations"]))
        for hop, ops_ in zip(hopSet_, opsSet_):
            shortestPaths = [
                path
                for path in shortestPathStrings
                if len(path) >= len(hop) and all(p == h for p, h in zip(path[: len(hop)], hop))
            ]
            shortestPathScore = len(shortestPaths)
            hasCapacity = bool(self.pathFreeCapacity(hop))
            score = (
                -np.inf
                if (not ops_[0].isApplicable) or (not hasCapacity)
                else shortestPathScore
            )
            freeCapacity = max([self.pathFreeCapacity(path) for path in shortestPaths], default=0)
            if score > bestScore or (score == bestScore and freeCapacity > bestFreeCapacity):
                bestScore = score
                bestOps = ops_
                bestFreeCapacity = freeCapacity
        return bestOps

    def refreshGraph(self) -> None:
        g = nx.DiGraph()

        for j in self._junctions:
            j.subgraph(g)
            j.numIons = len(j.ions)

        for t in self._manipulationTraps:
            t.subgraph(g)
            t.numIons = len(t.ions)

        for trap in self._manipulationTraps:
            trapEdges = []
            for ion1 in trap.ions:
                g.add_edge(ion1.idx, trap.idx, operations=[])
                for ion2 in trap.ions:
                    if ion1 == ion2:
                        continue
                    trapEdges.append((ion1.idx, ion2.idx))
                    trapEdges.append((ion2.idx, ion1.idx))
                    g.add_edge(
                        ion1.idx,
                        ion2.idx,
                        operations=[GateSwap.physicalOperation(trap=trap, ion1=ion1, ion2=ion2)],
                    )
                    g.add_edge(
                        ion2.idx,
                        ion1.idx,
                        operations=[GateSwap.physicalOperation(trap=trap, ion1=ion2, ion2=ion1)],
                    )
            self._trapEdges[trap.idx] = trapEdges

        crossingEdges = {}
        for crossing in self._crossings:
            n1, n2 = crossing.connection
            n1Idx = crossing.getEdgeIon(n1).idx if n1.ions else n1.idx
            n2Idx = crossing.getEdgeIon(n2).idx if n2.ions else n2.idx
            crossingEdges[(n1Idx, n2Idx)] = crossing
            crossingEdges[(n2Idx, n1Idx)] = crossing
            ion1 = crossing.getEdgeIon(n1) if n1.ions else None
            ion2 = crossing.getEdgeIon(n2) if n2.ions else None
            doRotation1 = [GateSwap.physicalOperation(trap=n1,ion1=ion1,ion2=ion1)] if len(n1.ions)==1 else []
            doRotation2 = [GateSwap.physicalOperation(trap=n2,ion1=ion2,ion2=ion2)] if len(n2.ions)==1 else []
            if isinstance(n1, Trap) and isinstance(n2, Junction):
                ops1 = doRotation1+[
                    Split.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    JunctionCrossing.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = [
                    JunctionCrossing.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    Merge.physicalOperation(n1, crossing, ion2),
                ]
            elif isinstance(n1, Junction) and isinstance(n2, Trap):
                ops1 = [
                    JunctionCrossing.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    Merge.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = doRotation2+[
                    Split.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    JunctionCrossing.physicalOperation(n1, crossing, ion2),
                ]
            elif isinstance(n1, Junction) and isinstance(n2, Junction):
                ops1 = [
                    JunctionCrossing.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    JunctionCrossing.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = [
                    JunctionCrossing.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    JunctionCrossing.physicalOperation(n1, crossing, ion2),
                ]
            else:
                ops1 = doRotation1+[
                    Split.physicalOperation(n1, crossing, ion1),
                    Move.physicalOperation(crossing, ion1),
                    Merge.physicalOperation(n2, crossing, ion1),
                ]
                ops2 = doRotation2+[
                    Split.physicalOperation(n2, crossing, ion2),
                    Move.physicalOperation(crossing, ion2),
                    Merge.physicalOperation(n1, crossing, ion2),
                ]
            g.add_edge(n1Idx, n2Idx, operations=ops1)
            g.add_edge(n2Idx, n1Idx, operations=ops2)
            if crossing.ion:
                g.add_node(crossing.ion.idx, pos=crossing.ion.pos)
        self._crossingEdges = crossingEdges

        for n2Idx in self._inActiveEdges:
            graphEdges = [
                (u, v)
                for (u, v), crossing in self._crossingEdges.items()
                if self.nodes[n2Idx] in crossing.connection
                and (v in self.nodes[n2Idx].nodes)
            ]
            g.remove_edges_from(graphEdges)
        self._graph = g
        self._centralities = None
        self._routingTable = {ion.idx: {} for ion in self.ions.values()}

    def display(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        title: str = "",
        operation: Optional[Operation] = None,
        show_junction: bool = True,
        showEdges: bool = True,
        showIons: bool = True,
        showLabels: bool = True,
        runOps: bool = False,
    ) -> None:
        pos = {}
        labels = {}
        operationNodes: List[List[int]] = []
        involvedIons: List[Sequence[Ion]] = []

        if operation is None:
            operations = []
        elif isinstance(operation, ParallelOperation):
            operations = operation.operations
            if runOps:
                operation.run()
                self.refreshGraph()
        else:
            operations = [operation]
            if runOps:
                for op in operations:
                    op.run()

                self.refreshGraph()

        for op in operations:
            operationNodes.append([])
            involvedIons.append(op.involvedIonsForLabel)

        for junction in self._junctions:
            pos[junction.nodes[0]] = junction.pos
            labels[junction.nodes[0]] = ""
            if show_junction:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=[junction.nodes[0]],
                    node_color=[junction.color],
                    node_shape=self.JUNCTION_SHAPE,
                    node_size=self.JUNCTION_SIZE,
                )
            for n, ion in zip(junction.nodes[1:], junction.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[n],
                        node_color=[ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showLabels:
                x = junction.pos[0]
                y = junction.pos[1]
                ax.text(
                    x,
                    y,
                    junction.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        for c in self._crossings:
            if c.ion:
                pos[c.ion.idx] = c.ion.pos
                labels[c.ion.idx] = c.ion.label
                if showIons:
                    nx.draw_networkx_nodes(
                        self._graph,
                        pos,
                        ax=ax,
                        nodelist=[c.ion.idx],
                        node_color=[c.ion.color],
                        node_shape=self.ION_SHAPE,
                        node_size=self.ION_SIZE,
                    )
                for nodes, ions in zip(operationNodes, involvedIons):
                    if c.ion in ions:
                        nodes.append(c.ion.idx)

        for t in self._manipulationTraps:
            if not isinstance(t, Trap):
                continue
            pos[t.nodes[0]] = t.pos
            labels[t.nodes[0]] = ""
            colors = {}
            for n, ion in zip(t.nodes[1:], t.ions):
                pos[n] = ion.pos
                labels[n] = ion.label
                colors[n] = ion.color
                for nodes, ions in zip(operationNodes, involvedIons):
                    if ion in ions:
                        nodes.append(n)
            if showIons:
                nx.draw_networkx_nodes(
                    self._graph,
                    pos,
                    ax=ax,
                    nodelist=t.nodes[1:],
                    node_color=colors.values(),
                    node_shape=self.ION_SHAPE,
                    node_size=self.ION_SIZE,
                )

        for trap in self._manipulationTraps:
            if not isinstance(trap, Trap):
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=trap[0],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color='red',
                    width=trap[1],
                )
                continue
            if showIons:
                nx.draw_networkx_edges(
                    self._graph,
                    pos,
                    edgelist=self._trapEdges[trap.idx],
                    ax=ax,
                    alpha=self.DEFAULT_ALPHA,
                    edge_color=trap.color,
                    width=self.TRAP_WIDTH,
                )
            if showLabels:
                x = trap.pos[0]
                y = trap.pos[1]
                ax.text(
                    x,
                    y,
                    trap.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showEdges:
            nx.draw_networkx_edges(
                self._graph,
                pos,
                edgelist=self._crossingEdges.keys(),
                ax=ax,
                alpha=self.DEFAULT_ALPHA,
                width=self.EDGE_WIDTH,
            )
        if showLabels:
            for e in self._crossings:
                ax.text(
                    *e.pos,
                    e.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        if showIons:
            nx.draw_networkx_labels(
                self._graph, pos, ax=ax, labels=labels, font_size=self.FONT_SIZE
            )

        for nodes, op in zip(operationNodes, operations):
            if nodes:
                xVals = [pos[node][0] for node in nodes]
                yVals = [pos[node][1] for node in nodes]
                padding = self.SIZING * self.PADDING
                xMin, xMax = min(xVals) - padding, max(xVals) + padding
                yMin, yMax = min(yVals) - padding, max(yVals) + padding
                width = xMax - xMin
                height = yMax - yMin
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ellip = Ellipse(
                    (xLabel, yLabel),
                    width,
                    height,
                    edgecolor=op.color,
                    alpha=self.DEFAULT_ALPHA,
                    facecolor=op.color,
                )
                ax.add_patch(ellip)
                xLabel = (xMin + xMax) / 2
                yLabel = (yMin + yMax) / 2
                ax.text(
                    xLabel,
                    yLabel,
                    op.label,
                    fontsize=self.FONT_SIZE,
                    bbox=dict(facecolor=self.HIGHLIGHT_COLOR, alpha=self.DEFAULT_ALPHA),
                )

        ax.set_title(title, fontsize=self.FONT_SIZE*5)
        n = len(fig.axes)
        fig.set_size_inches(self.WINDOW_SIZE[0]*n, self.WINDOW_SIZE[1])

    def _routingForQubit(
        self, operation: TwoQubitMSGate
    ) -> Tuple[Trap, Sequence[Operations], Sequence[Ion]]:
        ionsInvolved = list(operation.ions)
        ion1, ion2 = operation.ions
        trap = self.decideDestinationTrap(ion1, ion2)
        ion2.parent.numIons -= 1
        ion1.parent.numIons -= 1
        trap.numIons += 2
        movements = list(self.route(ion1, trap)) + list(self.route(ion2, trap))
        for m in movements:
            if isinstance(m, CrystalOperation):
                ionsInvolved += list(m.ionsInfluenced)
            m.addComponent(ion1)
            m.addComponent(ion2)
        return trap, movements, ionsInvolved

    def processOperationsViaRouting(
        self, operations: Sequence[QubitOperation]
    ) -> Sequence[Operation]:
        instructionsLeft = list(operations).copy()
        allOps = []
        while instructionsLeft:
            self.refreshGraph()
            physicalInstructions = []
            ionsInvolved = set()
            missedTwoQubitGate = False
            toRemove = []
            for op in instructionsLeft:
                trap = op.getTrapForIons()
                ionsInOp = list(op.ions)
                movements = []
                if not trap:
                    trap, movements, ionsInOp = self._routingForQubit(op)
                # order of CNOTs matter
                if ionsInvolved.isdisjoint(ionsInOp) and not (missedTwoQubitGate and isinstance(op,TwoQubitMSGate)):
                    physicalInstructions.extend(movements)
                    op.setTrap(trap)
                    physicalInstructions.append(op)
                    toRemove.append(op)
                elif isinstance(op, TwoQubitMSGate):
                    missedTwoQubitGate = True
                ionsInvolved = ionsInvolved.union(ionsInOp)
            for op in toRemove:
                instructionsLeft.remove(op)
            for op in physicalInstructions:
                if isinstance(op, GateSwap) and op._ion1==op._ion2:
                    continue
                op.run()
                allOps.append(op)
        return allOps

    def processOperationsViaForwarding(
        self,
        operations: Sequence[QubitOperation],
        opBarriers: Sequence[int] = [],
        goBack: bool = False
    ) -> Tuple[Sequence[Operation], Sequence[int]]:
        # predictedMovements: Dict[int, Dict[TwoQubitMSGate, Trap]] = {ion: {} for ion in self.ions}
        # twoqubitGatesForAncillaIons: Dict[int, List[TwoQubitMSGate]] = {ion: [] for ion in self.ions}
        # for op in operations:
        #     if isinstance(op, TwoQubitMSGate):
        #         ion1, ion2 = op.ions
        #         trap = self.decideDestinationTrap(ion1, ion2)
        #         ancilla, data = (ion1,ion2) if ion2.parent==trap else (ion2,ion1)
        #         twoqubitGatesForAncillaIons[ancilla.idx].append(op)
        #         predictedMovements[data.idx][op] = trap

        # for ion in self.ions:
        #     if not twoqubitGatesForAncillaIons[ion]:
        #         continue
        #     for o1, o2 in zip(twoqubitGatesForAncillaIons[ion][:-1], twoqubitGatesForAncillaIons[ion][1:]):
        #         predictedMovements[ion][o1] = self.decideDestinationTrap(o2.ions[0], o2.ions[1])
        #     predictedMovements[ion][twoqubitGatesForAncillaIons[ion][-1]] = self._originalArrangement[self.ions[ion]]

        self.refreshGraph()
        # Higher means less parallelism but fewer movements
        RANDOMIZATION_PROBABILITY = 1 / 5

        allOps = []
        
        opIdxs = {o: i for i, o in enumerate(operations)}
        operationsLeft = list(operations)
        idxsLeft = list([i for i in range(len(operations))])
        opBarriersLeft = list(opBarriers)
        nextBarrier = min(opBarriersLeft, default=len(operations))

        prevMovementScores: Mapping[Tuple[int, int], int] = {}
        movementScores: Mapping[Tuple[int, int], int] = {}
        toRemove: List[QubitOperation] = []
        requiresMovement: List[TwoQubitMSGate] = []
        if goBack:
            requiresMovementBack: List[TwoQubitMSGate] = []
        ionBlocks: List[Ion] = []
        barriers: List[int] =  []

        ionsInvolved: Set[Ion] = set()

        while operationsLeft:
            maxMovementScore = 0
            missedTwoQubitGate = False
            if goBack and requiresMovementBack:
                ionsInvolved = ionsInvolved.union(ionBlocks)
            elif goBack and len(ionBlocks)>0:
                ionBlocks = []
            #     barriers.append(len(allOps))
            
            if idxsLeft[0] < nextBarrier:
                for op, opIdx in zip(operationsLeft, idxsLeft):
                    if opIdx >= nextBarrier:
                        break
                    trap = op.getTrapForIons()
                    # Order of CNOTs matter 
                    if ionsInvolved.isdisjoint(op.ions) and not (missedTwoQubitGate and isinstance(op,TwoQubitMSGate)):
                        if (not trap)  or (goBack and op in requiresMovementBack): 
                            requiresMovement.append(op)
                            if not isinstance(op, TwoQubitMSGate):
                                raise ValueError(
                                    f"runOperationsViaForwarding: {op.label} is not a TwoQubitMsGate"
                                )
                            movementScores[op.ionsActedIdxs] = 1 + (
                                prevMovementScores[op.ionsActedIdxs]
                                if op.ionsActedIdxs in prevMovementScores
                                else 0
                            )
                        else:
                            op.setTrap(trap)
                            toRemove.append(op)
                    elif isinstance(op, TwoQubitMSGate):
                        missedTwoQubitGate = True and (len(opBarriers)==0)
                    ionsInvolved = ionsInvolved.union(op.ions)
            else:
                barriers.append(len(allOps))
                opBarriersLeft.remove(nextBarrier)
                nextBarrier = min(opBarriersLeft, default=len(operations))
                
            for op in toRemove:
                op.run()
                if goBack and isinstance(op, TwoQubitMSGate):
                    requiresMovementBack.append(op)
                else:
                    operationsLeft.remove(op)
                    idxsLeft.remove(opIdxs[op])

            allOps.extend(toRemove)
            if requiresMovement:
                maxMovementScore = max(movementScores.values())
                numAtMaxMovementScore = 0
                for k, score in movementScores.items():
                    numAtMaxMovementScore += score == maxMovementScore
                    if (numAtMaxMovementScore > 1) and (score == maxMovementScore):
                        movementScores[k] += np.random.choice(
                            [0, 1 / numAtMaxMovementScore],
                            p=[
                                1 - RANDOMIZATION_PROBABILITY,
                                RANDOMIZATION_PROBABILITY,
                            ],
                        )

                maxMovementScore = max(movementScores.values())
                priorityMovements = set(
                    [
                        op
                        for op in requiresMovement
                        if movementScores[op.ionsActedIdxs] == maxMovementScore
                    ]
                )
                inJunctionsMovements = set(
                    [op for op in requiresMovement if op.ionsInJunctions]
                )

                ionsInvolved.clear()
                for op in list(
                    inJunctionsMovements.difference(priorityMovements)
                ) + list(priorityMovements):
                    if not ionsInvolved.isdisjoint(op.ions):
                        continue

                    if goBack and op in requiresMovementBack:
                        hasMovements = False
                        for ion in op.ions:
                            trap = self._originalArrangement[ion]
                            movements = list(self.forward(ion, trap))
                            for m in movements:
                                if isinstance(m, CrystalOperation):
                                    ionsInvolved = ionsInvolved.union(m.ionsInfluenced)
                                if isinstance(m, GateSwap) and m._ion1==m._ion2:
                                    continue
                                m.run()
                                allOps.append(m)
                            if movements:
                                hasMovements = True
                                self.refreshGraph()
                        if not hasMovements:
                            requiresMovementBack.remove(op)
                            operationsLeft.remove(op)
                            ionBlocks.extend(op.ions)
                            idxsLeft.remove(opIdxs[op])
                    else:
                        ion1, ion2 = op.ions
                        trap = self.decideDestinationTrap(ion1, ion2)
                        for ion in op.ions:
                            movements = list(self.forward(ion, trap))
                            for m in movements:
                                if isinstance(m, CrystalOperation):
                                    ionsInvolved = ionsInvolved.union(m.ionsInfluenced)
                                if isinstance(m, GateSwap) and m._ion1==m._ion2:
                                    continue
                                m.run()
                                allOps.append(m)
                            if movements:
                                self.refreshGraph()
            prevMovementScores = movementScores
            movementScores = {}
            toRemove.clear()
            ionsInvolved.clear()
            requiresMovement.clear()
        return allOps, barriers
    

    def processOperationsWithSafety(
        self,
        operations: Sequence[QubitOperation],
        trapCapacity: int
    ) -> Tuple[Sequence[Operation], Sequence[int]]:
        twoqubitGatesForAncillaIons: Dict[int, List[TwoQubitMSGate]] = {}
        for op in operations:
            if isinstance(op, TwoQubitMSGate):
                ion1, ion2 = op.ions
                trap = self.decideDestinationTrap(ion1, ion2)
                ancilla, data = sorted(
                    (ion1, ion2), key=lambda ion: ion.label[0]=='D'
                )
                if ancilla.idx in twoqubitGatesForAncillaIons:
                    twoqubitGatesForAncillaIons[ancilla.idx].append(op)
                else:
                    twoqubitGatesForAncillaIons[ancilla.idx] = [op]
       
        opPriorities: Dict[Operation, int] = {op: i for i, op in enumerate(operations)}

        allOps: List[Operation] = []
        barriers: List[int] = []
        operationsLeft = list(operations)
        toMoveCandidates: Dict[int, TwoQubitMSGate] = {}
    
        while operationsLeft:


            # Run the operations that do not need routing
            
            while True:
                toRemove: List[Operation] = []
                ionsInvolved: Set[Ion] = set()
                for op in operationsLeft:
                    trap = op.getTrapForIons()
                    if ionsInvolved.isdisjoint(op.ions) and trap:
                        op.setTrap(trap)
                        toRemove.append(op)
                    ionsInvolved = ionsInvolved.union(op.ions)

                for op in toRemove:
                    op.run()
                    allOps.append(op)
                    operationsLeft.remove(op)
                    if isinstance(op, TwoQubitMSGate):
                        ion1, ion2 = op.ions
                        if ion1.idx in twoqubitGatesForAncillaIons:
                            twoqubitGatesForAncillaIons[ion1.idx].remove(op)
                        else:
                            twoqubitGatesForAncillaIons[ion2.idx].remove(op)
                
                if len(toRemove) == 0:
                    break

            # Determine the operations that need routing
            for ancillaIdx in twoqubitGatesForAncillaIons.keys():
                if ancillaIdx in toMoveCandidates:
                    continue
                if len(twoqubitGatesForAncillaIons[ancillaIdx]) == 0:
                    continue
                gate = twoqubitGatesForAncillaIons[ancillaIdx][0]
                trap = gate.getTrapForIons()
                if trap:
                    continue
                toMoveCandidates[ancillaIdx] = twoqubitGatesForAncillaIons[ancillaIdx].pop(0)

            # move operations with priority according to the original happens before
            toMove = sorted([(k,o) for k,o in toMoveCandidates.items()], key=lambda ko: opPriorities[ko[1]])

            crossingsUsed: Set[Crossing] = set()
            qccdNodesFull: Set[QCCDNode] = set()
            # ancillaIdx: op, pathChosen, destTrap, goBack
            movements: Dict[int, Tuple[TwoQubitMSGate, List[QCCDNode], Trap]] = {}


            # ionsInvolved = set()
            for ancillaIdx, op in toMove:
                ion1, ion2 = op.ions

                ancilla, data = (ion1, ion2) if ion1.idx == ancillaIdx else (ion2, ion1)
                trap = data.parent
                if not isinstance(trap, Trap):
                    raise ValueError(f"Data Ion not in a trap {trap}")
                
                src = ancillaIdx
                dest = trap.idx
                paths = list(nx.all_shortest_paths(self._graph, src, dest))
                
                qccdNodesChosen: List[QCCDNode] = []
                crossingsChosen: List[Crossing] = []
    
                for path in paths:
                    crossingsInPath: List[Crossing] = []
                    for n1, n2 in zip(path[:-1], path[1:]):
                        if (n1, n2) not in self._crossingEdges:
                            continue
                        crossingsInPath.append(self._crossingEdges[(n1,n2)])

                    qccdNodesInPath: List[QCCDNode] = []
                    for n in path: 
                        nd = self.nodes[n] if n in self.nodes else self.ions[n].parent
                        if nd not in qccdNodesInPath:
                            qccdNodesInPath.append(nd)   

                    qccdNodesInPathFull: List[QCCDNode] = []
                    # Do not include source since the source is going to decrease in ions or stay the same at all points
                    for qccdNode in qccdNodesInPath[1:]:
                        if isinstance(qccdNode, Junction) and qccdNode.numIons==1:
                            qccdNodesInPathFull.append(qccdNode)      
                        elif qccdNode.numIons == trapCapacity:
                            qccdNodesInPathFull.append(qccdNode)        
                               
                    if crossingsUsed.isdisjoint(crossingsInPath) and qccdNodesFull.isdisjoint(qccdNodesInPathFull):
                        qccdNodesChosen = qccdNodesInPath
                        crossingsChosen = crossingsInPath
                        break 

                # unable to complete move operation this time round
                if len(qccdNodesChosen) == 0:
                    continue 

                # able to complete move operation 
                toMoveCandidates.pop(ancillaIdx)
                movements[ancillaIdx]=(op, qccdNodesChosen, trap)
                # remove crossings that are reserved
                crossingsUsed = crossingsUsed.union(crossingsChosen)
                # increment traps and junctions number of ions by 1 EXCEPT the source
                # remove traps and junctions if currently at capacity EXCEPT the source 
                for qccdNode in qccdNodesChosen[1:]:
                    qccdNode.numIons+=1
                    if isinstance(qccdNode, Junction) and qccdNode.numIons==1:
                        qccdNodesFull.add(qccdNode)
                    elif qccdNode.numIons == trapCapacity:
                        qccdNodesFull.add(qccdNode)


            # if destination trap is at capacity then we need to send the ancilla back to original trap at start of barrier to maintain invariant
            toForward: Dict[TwoQubitMSGate, Tuple[int, List[QCCDNode], Trap, Optional[Trap]]] = {}
            for ancillaIdx, (op, qccdNodes, destTrap) in movements.items():
                if destTrap.numIons == trapCapacity:
                    goBackTrap=qccdNodes[0]
                    for nd in qccdNodes[::-1][:-1]:
                        # note we have already reserved the goBackTrap (qccdNode.numIons+=1) so no need to increment again
                        if nd.numIons <= trapCapacity-1 and isinstance(nd, Trap):
                            goBackTrap = nd
                    destTrap.numIons -= 1
                    # no need to increment srcTrap.numIons because we never decremented it in the first place
                else: 
                    goBackTrap=None
                toForward[op] = (ancillaIdx, qccdNodes, goBackTrap)

            startedGoingBack = {op: False for op in toForward.keys()}
            while toForward:
                ionsInvolved = set()
                orderedToForward = sorted([(o, rc) for o, rc in toForward.items()], key=lambda orc: opPriorities[orc[0]])
                for op, (ancillaIdx, qccdNodes, goBackTrap) in orderedToForward:
                    if not ionsInvolved.isdisjoint(op.ions):
                        continue 

                    ionsInvolvedNow = [self.ions[ancillaIdx]]

                    n1 = ancillaIdx
                    n1Idx = self.ions[ancillaIdx].parent.idx
                    trap = qccdNodes[-1]
                    srcTrap = goBackTrap
                
                    if n1Idx == trap.idx and not startedGoingBack[op]:
                        op.setTrap(trap)
                        op.run()
                        allOps.append(op)
                        operationsLeft.remove(op)
                        ionsInvolved = ionsInvolved.union(op.ions)
                        if goBackTrap is not None:
                            startedGoingBack[op] = True
                        else:
                            toForward.pop(op)
                            continue
                    elif startedGoingBack[op] and n1Idx == srcTrap.idx:
                        toForward.pop(op)
                        continue


                    if startedGoingBack[op]:
                        n2Idx = [dn.idx for sn, dn in zip(qccdNodes[::-1][:-1], qccdNodes[::-1][1:]) if sn.idx == n1Idx][0]
                    else:
                        n2Idx = [dn.idx for sn, dn in zip(qccdNodes[:-1], qccdNodes[1:]) if sn.idx == n1Idx][0]
                    forwardingPath = nx.shortest_path(self._graph, n1, n2Idx)
                    ms: List[Operation] = []
                    for n1, n2 in zip(forwardingPath[:-1], forwardingPath[1:]):
                        ms.extend(self._graph.edges[n1, n2]["operations"])
                        
                    for m in ms:
                        if isinstance(m, CrystalOperation):
                            ionsInvolvedNow.extend(m.ionsInfluenced)
                        # Hack FIXME
                        if isinstance(m, GateSwap) and m._ion1==m._ion2:
                            continue
                        m.run()
                        allOps.append(m)
                    
                    self.refreshGraph()
                    ionsInvolved = ionsInvolved.union(ionsInvolvedNow)

            barriers.append(len(allOps))

        return allOps, barriers


