
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree, distance_matrix
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist 
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
from qccd_arch import *
from qccd_parallelisation import *

class QCCDCircuit(stim.Circuit):
    DATA_QUBIT_COLOR = "lightblue"
    MEASUREMENT_QUBIT_COLOR = "red"
    PLACEMENT_ION = ("grey", "P")
    TRAP_COLOR = "grey"
    JUNCTION_COLOR = "orange"
    SPACING = 20


    start_score: int = 1
    score_delta: int = 2
    joinDisjointClusters: bool = False
    minIters: int = 1_000
    maxIters: int = 10_000


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ionMapping: Dict[int, Tuple[Ion, Tuple[int, int]]] = {}
        self._measurementIons: List[Ion] = []
        self._dataIons: List[Ion] = []
        self._originalArrangement: Dict[Trap, Sequence[Ion]] = {}
        self._arch: QCCDArch

    @classmethod
    def generated(cls, *args, **kwargs) -> "QCCDCircuit":
        return QCCDCircuit(stim.Circuit.generated(*args, **kwargs).__str__())

    def circuitString(self, include_annotation: bool = False) -> Sequence[str]:
        instructions = (
            self.flattened().decomposed().without_noise().__str__().splitlines()
        )
        newInstructions = []
        for i in instructions:
            qubits = i.rsplit(" ")[1:]
            if i.startswith("DETECTOR") or i.startswith("TICK") or i.startswith("OBSERVABLE"):
                if include_annotation:
                    newInstructions.append(i)
                continue
            elif i[0] in ("R", "H", "M"):
                for qubit in qubits:
                    newInstructions.append(f"{i[0]} {qubit}")
                # newInstructions.append("BARRIER")
            elif any(i.startswith(s) for s in stim.gate_data("cnot").aliases):
                for i in range(int(len(qubits) / 2)):
                    newInstructions.append(f"CNOT {qubits[2*i]} {qubits[2*i+1]}")
                newInstructions.append("BARRIER")
            else:
                newInstructions.append(i)
        return newInstructions

    @property
    def ionMapping(self) -> Mapping[int, Tuple[Ion, Tuple[int, int]]]:
        return self._ionMapping

    def _parseCircuitString(self, dataQubitsIdxs: Optional[Sequence[int]]=None) -> Tuple[Sequence[QubitOperation], Sequence[int]]:
        instructions = self.circuitString()

        self._measurementIons = []
        self._ionMapping = {}
        self._dataIons = []
        for j, i in enumerate(instructions):
            if not i.startswith("QUBIT_COORDS"):
                break
            coords = tuple(
                map(int, i.removeprefix("QUBIT_COORDS(").split(")")[0].split(","))
            )
            idx = int(i.split(" ")[-1])
            ion = QubitIon(self.MEASUREMENT_QUBIT_COLOR, label="M")
            ion.set(ion.idx, *coords)
            self._ionMapping[idx] = ion, coords
            self._measurementIons.append(ion)

        instructions = instructions[j:]
        operations = []
        barriers = []
        dataQubits = []
        # TODO establish correct mapping of qubit operations from QIP toolkit with references
        for j, i in enumerate(instructions):
            if i.startswith("BARRIER"):
                barriers.append(len(operations))
                continue
            if not ( i[0] in ("M", "H", "R") or i.startswith("CNOT")):
                continue
            idx = int(i.split(" ")[1])
            ion = self._ionMapping[idx][0]
            if i[0] == "M":
                operations.append(Measurement.qubitOperation(ion))
                if dataQubitsIdxs is None:
                    dataQubits.append(ion) # data qubits are the ones measured at the end
            elif i[0] == "H":
                # page 80 https://iontrap.umd.edu/wp-content/uploads/2013/10/FiggattThesis.pdf
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion)
                ])
                if dataQubitsIdxs is None:
                    dataQubits.clear()
            elif i[0] == "R":
                operations.append(QubitReset.qubitOperation(ion))
                if dataQubitsIdxs is None:
                    dataQubits.clear()
            elif i.startswith("CNOT"):
                idx2 = int(i.split(" ")[2])
                ion2 = self._ionMapping[idx2][0]
                # Fig 4. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.99.022330
                operations.extend([
                    YRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion),
                    XRotation.qubitOperation(ion2),
                    TwoQubitMSGate.qubitOperation(
                        ion, ion2
                    ),
                    YRotation.qubitOperation(ion)
                ])
                if dataQubitsIdxs is None:
                    dataQubits.clear()
        if dataQubitsIdxs is not None:
            dataQubits = [self._ionMapping[j][0] for j in dataQubitsIdxs]
        # TODO use cooling ions? probs not here since architecture dependent
        for d in dataQubits:
            d._color = self.DATA_QUBIT_COLOR
            d._label = "D"
            self._dataIons.append(d)
            self._measurementIons.remove(d)
        return operations, barriers
    
    def _clusterFromInstructions(
        self, ions: Sequence[Ion], instructions: Sequence[QubitOperation], trapCapacity: int, numClusters: int
    ) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]]:
        edgesDups = []
        ionsInvolved = set()
        score = self.start_score
        for op in instructions:
            if not isinstance(op, TwoQubitMSGate):
                continue
            if not ionsInvolved.isdisjoint(op.ions):
                score/=self.score_delta
                ionsInvolved=set()
            edgesDups.append((op.ions,score))
            ionsInvolved=ionsInvolved.union(op.ions)
        bestClusters: Sequence[List[Ion]] = [[] for _ in range(numClusters)]
        bestScoreTotal = np.inf
        # https://dl.acm.org/doi/pdf/10.1145/234533.234534 page 606 o(n**(2(k-1))) for k-way partition
        for num_iters in range(self.minIters+min(len(ions)**(2*(numClusters-1)), self.maxIters)):
            edges=edgesDups.copy()
            clusters = {ion: [ion] for ion in ions}
            scoreTotal=0
            while len(clusters)>numClusters:
                # https://dl.acm.org/doi/pdf/10.1145/234533.234534 page 611
                edgeProbabilities = np.array([s for _,s in edges])
                edgeProbabilities = edgeProbabilities/np.sum(edgeProbabilities)
                edgeChoices = np.random.choice(len(edges), p=edgeProbabilities, replace=False, size=len(edges))
                foundEdge=False
                for idx in edgeChoices:
                    ((ion1, ion2), s) = edges[idx]
                    if len(clusters[ion1])+len(clusters[ion2])<=trapCapacity:
                        edges.pop(idx)
                        foundEdge=True
                        break
                if not foundEdge:
                    if not self.joinDisjointClusters:
                        scoreTotal=np.inf 
                        break
                    ion1, ion2 = sorted(clusters, key=lambda ion: len(clusters[ion]))[0:2]
                clusters[ion1].extend(clusters.pop(ion2))
                newEdges = []
                for (i1, i2), s in edges:
                    i1 = i1 if i1 != ion2 else ion1
                    i2 = i2 if i2 != ion2 else ion1
                    if i1 != i2:
                        newEdges.append(((i1,i2), s))
                edges = newEdges
            scoreTotal+=sum(s for _, s in edges)
            if scoreTotal < bestScoreTotal:
                bestClusters = clusters.values()
                bestScoreTotal = scoreTotal
        
        if np.isinf(bestScoreTotal) and not self.joinDisjointClusters:
            edges=edgesDups.copy()
            clusters = {ion: [ion] for ion in ions}
            while len(clusters)>numClusters:
                foundEdge=False
                for idx, ((ion1, ion2), s) in enumerate(edges):
                    if len(clusters[ion1])+len(clusters[ion2])<=trapCapacity:
                        edges.pop(idx)
                        foundEdge=True
                        break
                if not foundEdge:
                    ion1, ion2 = sorted(clusters, key=lambda ion: len(clusters[ion]))[0:2]
                clusters[ion1].extend(clusters.pop(ion2))
                newEdges = []
                for (i1, i2), s in edges:
                    i1 = i1 if i1 != ion2 else ion1
                    i2 = i2 if i2 != ion2 else ion1
                    if i1 != i2:
                        newEdges.append(((i1,i2), s))
                edges = newEdges
            bestClusters = clusters.values()

        clusters = []
        for clusterIons in bestClusters:
            clusterCentre = np.mean([list(ion.pos) for ion in clusterIons], axis=0)
            clusters.append((clusterIons, clusterCentre))
        return clusters
        


    
    def _deprecatedClusterIons(
        self, ions: Sequence[Ion], coords: npt.NDArray[np.float_], trapCapacity: int
    ) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]]:
        unclusteredIndices = set(range(len(coords)))
        clusters = []
        while unclusteredIndices:
            currentClusterIndices = []
            currentClusterCoords = []

            # Choose the first unclustered ion as the starting point
            start_index = next(iter(unclusteredIndices))
            currentClusterIndices.append(start_index)
            currentClusterCoords.append(coords[start_index])

            unclusteredIndices.remove(start_index)

            # Keep adding ions to the cluster until it reaches the trapCapacity
            while len(currentClusterIndices) < trapCapacity and unclusteredIndices:
                # Find the ion closest to the current cluster's centroid
                cluster_centroid = np.mean(currentClusterCoords, axis=0)
                distances = distance.cdist(
                    [cluster_centroid], coords[list(unclusteredIndices)]
                )[0]
                nearest_index = list(unclusteredIndices)[np.argmin(distances)]
                # Add the nearest ion to the current cluster
                currentClusterIndices.append(nearest_index)
                currentClusterCoords.append(coords[nearest_index])
                unclusteredIndices.remove(nearest_index)

            clusterIons = [ions[i] for i in currentClusterIndices]
            clusterCentre = np.mean(currentClusterCoords, axis=0)
            clusters.append((clusterIons, clusterCentre))
        return clusters
    
    def _partitionClusterIons(
        self, ions: Sequence[Ion], coords: npt.NDArray[np.float_], trapCapacity: int
    ) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]]:
        partitions = [list(coords)]
        splitAxisIsX = True
        while max([len(p) for p in partitions])>trapCapacity:
            toSplit = [p for p in partitions if len(p)>trapCapacity]
            for p in toSplit:
                splitAxisVals = [float(c[int(splitAxisIsX)]) for c in p]
                medAxisVal = np.mean(splitAxisVals)
                p1 = []
                p2 = []
                for c, splAxisVal in zip(p, splitAxisVals):
                    if splAxisVal <= medAxisVal:
                        p1.append(c)
                    else:
                        p2.append(c)
                if p1:
                    partitions.append(p1)
                if p2:
                    partitions.append(p2)
            splitAxisIsX = not splitAxisIsX
            for p in toSplit:
                partitions.remove(p)

        coordsToIons = {(c[0], c[1]): i for c, i in zip(coords, ions)}
        clusters = []
        for p in partitions:
            clusterIons = [coordsToIons[(c[0], c[1])] for c in p]
            clusterCentre = np.mean(p, axis=0)
            clusters.append((clusterIons, clusterCentre))
        return clusters
    
    def _partitionClusterClusters(
        self, clusters: Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]], ancillaRatio: int
    ) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]]:
        partitions = [list(c[1] for c in clusters)]
        splitAxisIsX = True
        while max([len(p) for p in partitions])>ancillaRatio:
            toSplit = [p for p in partitions if len(p)>ancillaRatio]
            for p in toSplit:
                splitAxisVals = [float(c[int(splitAxisIsX)]) for c in p]
                medAxisVal = np.mean(splitAxisVals)
                p1 = []
                p2 = []
                for c, splAxisVal in zip(p, splitAxisVals):
                    if splAxisVal <= medAxisVal:
                        p1.append(c)
                    else:
                        p2.append(c)
                if p1:
                    partitions.append(p1)
                if p2:
                    partitions.append(p2)
            splitAxisIsX = not splitAxisIsX
            for p in toSplit:
                partitions.remove(p)

        clustersLeft = list(clusters)
        clusters = []
        for p in partitions:
            clusterIons = []
            clusterCentre = np.zeros_like(p[0])
            for c in p:
                cl = [cl for cl in clustersLeft if c[0]==cl[1][0] and c[1]==cl[1][1]][0]
                cIons = cl[0]
                clusterCentre += len(cIons)*c
                clusterIons+=cIons
                clustersLeft.remove(cl)
            clusterCentre = (1/len(clusterIons))*clusterCentre
            clusters.append((clusterIons, clusterCentre))
        return clusters


    def _clusterIons(
        self, measurementIons: Sequence[Ion], dataIons: Sequence[Ion], trapCapacity: int, edges: Sequence[Tuple[Ion, Ion]], maxClusters: int
    ) -> Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]]:

        measurementCoords = np.array([list(ion.pos) for ion in measurementIons])

        dataIonsForMeasurementIon = {measurementIon: [(ion1 if ion1!=measurementIon else ion2) for ion1, ion2 in edges if measurementIon in (ion1, ion2)] for measurementIon in measurementIons}
        dataIonsPerMeasurementIon = np.mean([len(dIons) for dIons in dataIonsForMeasurementIon.values()])

        measurementIonsPerCluster = max(round(trapCapacity / dataIonsPerMeasurementIon),1)
        numClusters = min(int(len(measurementIons)/measurementIonsPerCluster), maxClusters)

            
        unclusteredMIndices = set(range(len(measurementIons)))
        unclusteredDIndices = set(range(len(dataIons)))

        clustersMIndices = []
        clustersIons = []
        clustersCentre = []
        for _ in range(numClusters):
            currentClusterMIndices: List[int] = []
            currentClusterMCoords = []
            # Choose the first unclustered ion as the starting point
            start_index = next(iter(unclusteredMIndices))
            currentClusterMIndices.append(start_index)
            currentClusterMCoords.append(measurementCoords[start_index])

            unclusteredMIndices.remove(start_index)

            # Keep adding measurement ions to the cluster until it reaches the limit
            for _ in range(measurementIonsPerCluster-1):
                unclusteredMIndicesList = list(unclusteredMIndices)
                dataQubitsForCurrentCluster = []
                for mIndex in currentClusterMIndices:
                    dataQubitsForCurrentCluster.extend(dataIonsForMeasurementIon[measurementIons[mIndex]])
                dataQubitsForCurrentCluster = set(dataQubitsForCurrentCluster)
                nonSharedDataQubits = np.array([len(dataQubitsForCurrentCluster.symmetric_difference(dataIonsForMeasurementIon[measurementIons[mIndex]])) for mIndex in unclusteredMIndicesList])
                nearestIndices =np.array(unclusteredMIndicesList)[nonSharedDataQubits==min(nonSharedDataQubits)]

                # Find the ion closest to the current cluster's centroid
                clusterCentroid = np.mean(currentClusterMCoords, axis=0)
                distances = distance.cdist(
                    [clusterCentroid], measurementCoords[nearestIndices]
                )[0]
                nearestIndex = nearestIndices[np.argmin(distances)]
                # Add the nearest ion to the current cluster
                currentClusterMIndices.append(nearestIndex)
                currentClusterMCoords.append(measurementCoords[nearestIndex])
                unclusteredMIndices.remove(nearestIndex)

            clusterCentre = np.mean(currentClusterMCoords, axis=0)
            clusterIons = list(measurementIons[i] for i in currentClusterMIndices)
            
            clustersMIndices.append(currentClusterMIndices)
            clustersIons.append(clusterIons)
            clustersCentre.append(clusterCentre)

        for maxScore in list(range(measurementIonsPerCluster+1))[::-1]:
            if not unclusteredDIndices:
                break
            for currentClusterMIndices, clusterIons in zip(clustersMIndices, clustersIons):
                if not unclusteredDIndices:
                    break
                unclusteredDIndicesList = list(unclusteredDIndices)
                dataIonOrdering = np.array([sum([(dataIons[idx] in dataIonsForMeasurementIon[measurementIons[mIndex]]) for mIndex in currentClusterMIndices]) for idx in unclusteredDIndicesList])
                nearestIndices = np.array(unclusteredDIndicesList)[dataIonOrdering==maxScore] 
                nIdxs = trapCapacity-len(clusterIons)
                for idx in nearestIndices[:nIdxs]:
                    clusterIons.append(dataIons[idx])
                    unclusteredDIndices.remove(idx)
        
        clusters = list(zip(clustersIons, clustersCentre))
        freeClusters = [(clusterIons, clusterCentre) for clusterIons, clusterCentre in clusters if len(clusterIons)<trapCapacity]
        while unclusteredDIndices or unclusteredMIndices:
            if unclusteredDIndices:
                unclusteredIon = dataIons[unclusteredDIndices.pop()]
            else:
                unclusteredIon = measurementIons[unclusteredMIndices.pop()]
            if not freeClusters:
                cluster= ([unclusteredIon], np.array(unclusteredIon.pos))
                clusters.append(cluster)
                if trapCapacity>1:
                    freeClusters.append(cluster)
                continue
            distances = distance.cdist(
                [unclusteredIon.pos], [pos for _,pos in freeClusters]
            )[0]
            clusterIons, clusterCentre = freeClusters[np.argmin(distances)]
            clusterIons.append(unclusteredIon)
            if len(clusterIons)==trapCapacity:
                freeClusters.pop(np.argmin(distances))
        return clusters

    def _objectiveFunctionForArrangement(
        self,
        gridPos: npt.NDArray[np.int_],
        edges: Sequence[Tuple[int, int]],
        clusters: Sequence[Tuple[Sequence[Ion], npt.NDArray[np.float_]]],
    ):
        distances = []
        differences = []
        for u, v in edges:
            ionsInvolved = len(clusters[u][0]) * len(clusters[v][0])
            vectorBefore = clusters[u][1] - clusters[v][1]
            vectorBeforeSize = np.linalg.norm(vectorBefore)
            if vectorBeforeSize!=0:
                vectorBefore = vectorBefore / vectorBeforeSize
            vectorAfter = gridPos[u] - gridPos[v]
            vectorAfterSize = np.linalg.norm(vectorAfter)
            if vectorAfterSize!=0:
                vectorAfter = vectorAfter / vectorAfterSize
            dotProd = np.dot(vectorAfter, vectorBefore)
            if isinstance(dotProd, float) and -1 <= dotProd <= 1:
                cosDiff = np.arccos(dotProd)
            else:
                cosDiff = np.inf
            dist = np.linalg.norm(gridPos[u] - gridPos[v])
            distances.append(dist * ionsInvolved)
            differences.append(cosDiff * ionsInvolved)
        return (
            sum(distances)
            + sum(differences) 
            + ( np.inf if any(d == 0 for d in distances)  else 0)
        )

    def _arrangeClusters(
        self,
        clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
        rows: int,
        cols: int,
        n_iters: int = 5000,
    ):
        numClusters = len(clusters)
        edges = np.array(
            [[(i, j) for i in range(numClusters) if i != j] for j in range(numClusters)]
        ).reshape(((numClusters - 1) * numClusters, 2))

        # Only consider grid positions within a minimal region to avoid transport overheads
        rs = min(rows, int(1 + numClusters / cols))
        cs = cols
        allGridPos = []
        for c in range(cs):
            allGridPos+=[(c, r) for r in range(rs)]
        # Initial discrete grid positions
        initialGridPos = allGridPos[:numClusters].copy()
        freeGridPos = allGridPos[numClusters:].copy()

        bestPos = initialGridPos.copy()
        bestError = self._objectiveFunctionForArrangement(
            np.array(initialGridPos), edges, clusters
        )
        newPos = initialGridPos
        # FIXME: SEE BELOW FUNCTION Simple random walk optimisation algorithm, without hill climbing, perhaps reconsider
        for _ in range(n_iters):
            cluster = np.random.randint(numClusters)
            gridPos = np.random.randint(cs), np.random.randint(rs)
            oldGridPos = newPos[cluster]
            if gridPos in freeGridPos:
                freeGridPos[freeGridPos.index(gridPos)] = oldGridPos
            else:
                newPos[newPos.index(gridPos)] = oldGridPos
            newPos[cluster] = gridPos
            newError = self._objectiveFunctionForArrangement(np.array(newPos), edges, clusters)
            if newError < bestError:
                bestError = newError
                bestPos = newPos
        return bestPos

    def _arrangeClustersAugmentedGrid(
        self,
        clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
        rows: int,
        cols: int,
        allGridPos: Sequence[Tuple[int, int]],
        compact: bool = False
    ):
        A, B = np.array([c[1] for c in clusters]), np.array(allGridPos)
       
        # Function to iteratively compact points by shifting them towards the centroid
        def iterative_compacting(B, mapped_indices):
            mapped_points = B[mapped_indices]
            unused_indices = np.setdiff1d(np.arange(len(B)), mapped_indices)
            unused_points = B[unused_indices]
            
            for cutoff in (np.sqrt(2),2):
                # Compute the centroid of the current mapped points in B
                centroid = np.mean(mapped_points, axis=0)
                # Repeat the compacting process until no more points can move closer
                moved = True
                while moved:
                    moved = False
                    
                    # Compute distances from the centroid for mapped points
                    distances_from_centroid = np.linalg.norm(mapped_points - centroid, axis=1)
                    
                    # Sort mapped points by their distance to the centroid (closest first)
                    sorted_indices = np.argsort(-distances_from_centroid)
                    
                    for i in sorted_indices:
                        current_point = mapped_points[i]
                        unused_points_to_consider = unused_points[np.linalg.norm(unused_points - centroid, axis=1) < np.linalg.norm(current_point - centroid)]
                        distances_to_unused_to_consider = np.linalg.norm(unused_points_to_consider - current_point, axis=1)
                        unused_points_to_consider = unused_points_to_consider[distances_to_unused_to_consider<=cutoff]
                        distances_to_unused_to_consider = distances_to_unused_to_consider[distances_to_unused_to_consider<=cutoff]
                        if len(distances_to_unused_to_consider) == 0:
                            continue
                        
                        # Find the nearest unused point to the current point
                        nearest_unused_idx_to_consider = np.argmin(distances_to_unused_to_consider)
                        nearest_unused_point = unused_points_to_consider[nearest_unused_idx_to_consider]
                    
                        # Move the current point to the nearest unused point
                        # Update mapped_indices to reflect this reassignment
                        original_index = np.where(np.all(B == current_point, axis=1))[0][0]
                        new_index_inB = np.where(np.all(B == nearest_unused_point, axis=1))[0][0]
                        new_index_in_unused_points = np.where(np.all(unused_points == nearest_unused_point, axis=1))[0][0]
                        mapped_indices[i] = new_index_inB
                        mapped_points[i] = nearest_unused_point
                            
                        # Update unused_points and unused_indices
                        unused_points = np.delete(unused_points, new_index_in_unused_points, axis=0)
                        unused_indices = np.setdiff1d(unused_indices, [new_index_inB])
                        unused_points = np.vstack([unused_points, [B[original_index]]])
                        unused_indices = np.append(unused_indices, original_index)
                            
                        moved = True 
                
            # Return the compacted mapped indices
            return mapped_indices

        # Function to map points from A to B using global and compactness constraints
        def map_A_to_B_with_compactness(A, B):
            cost_matrix = distance_matrix(A, B)
            # the Hungarian algorithm to find the initial injective mapping
            _, col_ind = linear_sum_assignment(cost_matrix)
            return col_ind

        # Mapping points from A to B with a combination of global structure preservation and compactness
        mapping = map_A_to_B_with_compactness(A, B)
        if compact:
            mapping =iterative_compacting(B, mapping)

        return [allGridPos[idx] for idx in mapping]

    def _gridToCoordinate(
        self, pos: Tuple[int, int], trapCapacity: int
    ) -> npt.NDArray[np.float_]:
        return np.array(pos) * (trapCapacity + 1) * self.SPACING

    def resetArch(
        self
    ) -> QCCDArch:
        for node in self._arch.nodes.values():
            while node.ions:
                node.removeIon(node.ions[0])

        for crossing in self._arch._crossings:
            if crossing.ion:
                crossing.clearIon()

        for trap, ions in self._originalArrangement.items():
            for i, ion in enumerate(ions):
                trap.addIon(ion, offset=i)
                ion.addMotionalEnergy(-ion.motionalMode)
        return self._arch
    
    def simulate(self, operations: Sequence[Operation], num_shots: int = 100_000, error_scaling: float = 1.0, decode: bool = True) -> Tuple[float, float, float]:
        # TODO add the effect of dephasing noise from idling qubits involved in splits and merges into this simulation (see notability notes)
        # TODO add importance subset sampling (see notability notes)
        # TODO speed up with sinter (see stim/getting_started)
        stimInstructions = self.circuitString(include_annotation=True)
        
        stimIdxs: List[int] = []
        ions: List[Ion] = []
        for stimIdx, (ion, _) in self._ionMapping.items():
            stimIdxs.append(stimIdx)
            ions.append(ion)

        operationsForIons: Dict[int, List[QubitOperation]] = {stimIdx: [] for stimIdx in stimIdxs}
        gateSwapsForIons: Dict[int, List[Tuple[int, GateSwap]]] = {stimIdx: [] for stimIdx in stimIdxs}
        qubitOps = []
        for op in operations:
            if isinstance(op, GateSwap):
                ion = op.ions[0] # first ion is the source ion in the gate swap
                # assert ion.label[0] == "M", "only swap ancillas"
                stimIdxForIon = stimIdxs[ions.index(ion)]
                opForIonIdx = len(operationsForIons[stimIdxForIon])
                gateSwapsForIons[stimIdxForIon].append(( opForIonIdx, op))
            elif isinstance(op, QubitOperation):
                for ion in op.ions:
                    operationsForIons[stimIdxs[ions.index(ion)]].append(op)
                qubitOps.append(op)

        gateSwapsForOperations: Dict[QubitOperation, List[GateSwap]] = {op: [] for op in qubitOps}
        for stimIdx, gateSwaps in gateSwapsForIons.items():
            for (opForIonIdx, op) in gateSwaps:
                gateSwapsForOperations[operationsForIons[stimIdx][opForIonIdx]].append(op)

        meanPhysicalZError = 0.0
        meanPhysicalXError = 0.0
        dephasingSchedule  = calculateDephasingFromIdling(operations)
        dephasingSchedule = dict(dephasingSchedule)

        numZGates = 0
        numXGates = 0
        circuitString = ''
        for i in stimInstructions:
            if i.startswith("BARRIER"):
                continue
            idx = int(i.split(" ")[1]) if ( i[0] in ("M", "H", "R") or i.startswith("CNOT")) else -1
            doNoiseAfter = False if i[0]=="M" else True
            if i[0] == "M" or i[0] == "R":
                ops = operationsForIons[idx][:1]
                operationsForIons[idx].pop(0)
            elif i[0] == "H":
                ops = operationsForIons[idx][:2]
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
            elif i.startswith("CNOT"):
                idx2 = int(i.split(" ")[2])
                # Do not duplicate the two qubit gate
                ops = operationsForIons[idx][:3] + operationsForIons[idx2][:1]
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx].pop(0)
                operationsForIons[idx2].pop(0)
                operationsForIons[idx2].pop(0)
            else:
                ops = []

            physicalZError = 0.0
            physicalXError = 0.0
            for op in ops:
                if op in qubitOps:
                    for ion in op.ions:
                        if len(dephasingSchedule[ion])>0:
                            dephasing = [dephasingFidelity for opAtEndOfIdle, dephasingFidelity in dephasingSchedule[ion] if opAtEndOfIdle==op]
                            if dephasing:
                                dephasingInFidelity = min((1-dephasing[0])/error_scaling, 0.5)
                                physicalZError += dephasingInFidelity
                                circuitString+=f"Z_ERROR({dephasingInFidelity}) {stimIdxs[ions.index(ion)]}\n"
                    for gs in gateSwapsForOperations[op]:
                        gsInfidelity = min((1-gs.fidelity())/error_scaling, 0.5)
                        physicalXError += gsInfidelity/2
                        physicalZError += gsInfidelity/2
                        circuitString+=f"DEPOLARIZE2({gsInfidelity}) {stimIdxs[ions.index(gs.ions[0])]} {stimIdxs[ions.index(gs.ions[1])]}\n"
                      
                        
            if doNoiseAfter:
                circuitString+=f'{i}\n'

            for op in ops:
                opInfidelity = min((1-op.fidelity())/error_scaling, 0.5)
                if len(op.ions)==1: 
                    if isinstance(op, QubitReset) or isinstance(op, Measurement):
                        physicalXError+=opInfidelity
                        circuitString+=f"X_ERROR({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]}\n"
                    else:
                        physicalXError+=opInfidelity/2
                        physicalZError+=opInfidelity/2
                        circuitString+=f"DEPOLARIZE1({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]}\n"
                elif len(op.ions)==2:
                    physicalXError+=opInfidelity/2
                    physicalZError+=opInfidelity/2
                    circuitString+=f"DEPOLARIZE2({opInfidelity}) {stimIdxs[ions.index(op.ions[0])]} {stimIdxs[ions.index(op.ions[1])]}\n"
                else:
                    raise ValueError(f"simulate: {op} contains {len(op.ions)} ions.")
            numZGates+=physicalZError>0
            numXGates+=physicalXError>0
            meanPhysicalZError += physicalZError
            meanPhysicalXError += physicalXError
            if not doNoiseAfter:
                circuitString+=f'{i}\n'
        meanPhysicalZError /= numZGates
        meanPhysicalXError /= numXGates
        circuit = stim.Circuit(circuitString)
        if not decode:
            return 1, meanPhysicalXError, meanPhysicalZError
        # Sample the circuit, by using the fast circuit stabilizer tableau simulator provided by Stim.
        sampler = circuit.compile_detector_sampler()
        sample =sampler.sample(num_shots, separate_observables=True)
        detection_events, observable_flips = sample
        detection_events = np.array(detection_events, order='C')

        # Construct a Tanner graph, by translating the detector error model using the circuit.
        detector_error_model = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

        # Determine the predicted logical observable, by running the MWPM decoding algorithm on the Tanner graph
        predictions = []
        for i in range(num_shots):
            predictions.append(matcher.decode(detection_events[i]))
        predictions=np.array(predictions)

        # Count the mistakes.
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        logicalError = num_errors / num_shots 
        return logicalError, meanPhysicalXError, meanPhysicalZError

                


    # def processCircuit(
    #     self,
    #     trapCapacity: int = 2,
    #     rows: int = 1,
    #     cols: int = 5,
    #     isHorizontal: bool =False,
    #     clustering: int = 1,
    #     padding: int = 1,
    #     dataQubitIdxs: Optional[Sequence[int]]=None
    # ) -> Tuple[QCCDArch, Sequence[QubitOperation]]:        
    #     instructions = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
    #     if trapCapacity * rows * cols < len(self._ionMapping):
    #         raise ValueError("processCircuit: not enough traps")
    #     if clustering==1:
    #         edgesDups = [tuple(sorted(op.ions, key=lambda ion: ion.pos)) for op in instructions if isinstance(op, TwoQubitMSGate)]
    #         edges = set(edgesDups)

    #         clusters = self._clusterIons(self._measurementIons, self._dataIons, trapCapacity, edges, rows*cols)
    #     elif clustering==2:
    #         numClusters = int( len(self._ionMapping) / trapCapacity + (len(self._ionMapping) % trapCapacity>0))
    #         clusters = self._clusterFromInstructions(list(self._measurementIons)+list(self._dataIons), instructions, trapCapacity, numClusters)
    #     else:
    #         ions = list(self._measurementIons)+list(self._dataIons)
    #         ionCoords = np.array([list(ion.pos) for ion in ions])
    #         clusters=self._deprecatedClusterIons(ions, ionCoords, trapCapacity)
    #     gridPositions = self._arrangeClusters(clusters, rows, cols)

    #     gridPositions = [(c+padding, r+padding) for (c, r) in gridPositions]
            
    #     trap_for_grid = {
    #         (col, row): clusters[trapIdx]
    #         for trapIdx, (col, row) in enumerate(gridPositions)
    #     }
    #     self._originalArrangement = {}

    #     self._arch = QCCDArch()
    #     traps_dict = {}
    #     for row in range(rows+2*padding):
    #         for col in range(cols+2*padding):
    #             if (col, row) in trap_for_grid:
    #                 ions = trap_for_grid[(col, row)][0]
    #             else:
    #                 ions = []
    #             traps_dict[(col, row)] = self._arch.addManipulationTrap(
    #                 *self._gridToCoordinate((col, row), trapCapacity),
    #                 ions,
    #                 color=self.TRAP_COLOR,
    #                 isHorizontal=isHorizontal,
    #                 capacity=trapCapacity
    #             )
    #             self._originalArrangement[traps_dict[(col, row)]] = ions
            
    #     # TODO use honeycomb architecture aswell so split this class
    #     if rows == 1:
    #         for (col, r), trap_node in traps_dict.items():
    #             if (col + 1, r) in traps_dict:
    #                 self._arch.addEdge(trap_node, traps_dict[(col + 1, r)])
        
    #     else:
    #         junctions_dict = {}
    #         for (col, row), trap_node in traps_dict.items():
    #             # Add vertical edges (between rows)
    #             if (col, row + 1) in traps_dict:
    #                 junction = self._arch.addJunction(
    #                     *(
    #                         (
    #                             self._gridToCoordinate((col, row), trapCapacity)
    #                             + self._gridToCoordinate((col, row + 1), trapCapacity)
    #                         )
    #                         / 2
    #                     ),
    #                     color=self.JUNCTION_COLOR,
    #                 )
    #                 junctions_dict[(col, row)] = junction
    #                 self._arch.addEdge(trap_node, junction)
    #                 self._arch.addEdge(junction, traps_dict[(col, row + 1)])

    #         # Add horizontal edges between junctions in the same row
    #         for row in range(rows):
    #             for col in range(cols - 1):
    #                 if (col, row) in junctions_dict and (
    #                     col + 1,
    #                     row,
    #                 ) in junctions_dict:
    #                     self._arch.addEdge(
    #                         junctions_dict[(col, row)], junctions_dict[(col + 1, row)]
    #                     )

    #     return self._arch, instructions

    def processCircuitAugmentedGrid(
        self,
        trapCapacity: int = 2,
        rows: int = 1,
        cols: int = 5,
        padding: int = 1,
        dataQubitIdxs: Optional[Sequence[int]]=None
        # capacityIsInTermsOfDataIons: bool = False
    ) -> Tuple[QCCDArch, Tuple[Sequence[QubitOperation], Sequence[int]]]:        
        instructions, barriers = self._parseCircuitString(dataQubitsIdxs=dataQubitIdxs)
        if trapCapacity * ((rows-1) * (2*cols-1)+cols) < len(self._ionMapping):
            raise ValueError("processCircuit: not enough traps")
           

        dIonsPerTrap = trapCapacity 
        while True:
            measurementIons = list(self._measurementIons)
            measurementIonCoords = np.array([list(ion.pos) for ion in measurementIons])
            dataIons = list(self._dataIons)
            dataIonCoords = np.array([list(ion.pos) for ion in dataIons])
            clustersD=list(self._partitionClusterIons(dataIons, dataIonCoords, dIonsPerTrap)) 
            clustersM=list(self._partitionClusterIons(measurementIons, measurementIonCoords, 1)) 
            clusters = list(clustersD)
            for clusterM in clustersM:
                cl = min(clusters, key=lambda c: (c[1][0]-clusterM[1][0])**2+(c[1][1]-clusterM[1][1])**2)
                cIons = list(cl[0])+list(clusterM[0])
                rD = len(cl[0])/len(cIons)
                clusters.append((cIons, clusterM[1]*(1-rD)+cl[1]*rD))
                clusters.remove(cl)
            maxClusterSize = max([len(c[0]) for c in clusters])
            if maxClusterSize > trapCapacity-1:
                if dIonsPerTrap == 2:
                    ions = list(self._measurementIons)+list(self._dataIons)
                    ionCoords = np.array([list(ion.pos) for ion in ions])
                    clusters=self._partitionClusterIons(ions, ionCoords, trapCapacity-1)
                    break 
                dIonsPerTrap -= 1
            else:
                break


        cs, rs = cols, rows
        allGridPos = []
        for r in range(rs):
            for c in range(cs):
                allGridPos.append((2*c, 2*r))
                if c < cs-1 and r<rs-1:
                    allGridPos.append((2*c+1, 2*r+1)) 

        gridPositions = self._arrangeClustersAugmentedGrid(clusters, rows, cols, (trapCapacity>=2), allGridPos=allGridPos)
        gridPositions = [(c+padding, r+padding) for (c, r) in gridPositions]
        rows = rows+2*padding
        cols = cols+2*padding
        trap_for_grid = {
            (col, row): clusters[trapIdx]
            for trapIdx, (col, row) in enumerate(gridPositions)
        }
        self._originalArrangement = {}

        self._arch = QCCDArch()
        traps_dict = {}
        for row in range(rows):
            for col in range(cols):
                if (2*col, 2*row) in trap_for_grid:
                    ions = trap_for_grid[(2*col, 2*row)][0]
                else:
                    ions = []
                traps_dict[(2*col, 2*row)] = self._arch.addManipulationTrap(
                    *self._gridToCoordinate((2*col, 2*row), trapCapacity),
                    ions,
                    color=self.TRAP_COLOR,
                    isHorizontal=False,
                    capacity=trapCapacity
                )
                self._originalArrangement[traps_dict[(2*col, 2*row)]] = ions

            if row == rows-1:
                break

            for col in range(cols-1):
                if (2*col+1, 2*row+1) in trap_for_grid:
                    ions = trap_for_grid[(2*col+1, 2*row+1)][0]
                else:
                    ions = []
                traps_dict[(2*col+1, 2*row+1)] = self._arch.addManipulationTrap(
                    *self._gridToCoordinate((2*col+1, 2*row+1), trapCapacity),
                    ions,
                    color=self.TRAP_COLOR,
                    isHorizontal=True,
                    capacity=trapCapacity
                )
                self._originalArrangement[traps_dict[(2*col+1, 2*row+1)]] = ions
            
        # TODO use honeycomb architecture aswell so split this class
        if rows == 1:
            for (col, r), trap_node in traps_dict.items():
                if (col + 1, r) in traps_dict:
                    self._arch.addEdge(trap_node, traps_dict[(col + 1, r)])
        else:
            junctions_dict = {}
            for (col, row), trap_node in traps_dict.items():
                # Add vertical edges (between even rows)
                if col % 2 == 0 and (col, row + 2) in traps_dict:
                    junction = self._arch.addJunction(
                        *(
                            (
                                self._gridToCoordinate((col, row), trapCapacity)
                                + self._gridToCoordinate((col, row + 2), trapCapacity)
                            )
                            / 2
                        ),
                        color=self.JUNCTION_COLOR,
                    )
                    junctions_dict[(col, row+1)] = junction
                    self._arch.addEdge(trap_node, junction)
                    self._arch.addEdge(junction, traps_dict[(col, row + 2)])

            # Add horizontal edges between traps and junctions in the same row
            for row in range(rows-1):
                for col in range(cols - 1):
                    if (2*col, 2*row+1) in junctions_dict and (
                        2*col + 1,
                        2*row + 1,
                    ) in traps_dict:
                        self._arch.addEdge(
                            junctions_dict[(2*col, 2*row+1)], traps_dict[(2*col + 1, 2*row+1)]
                        )
                    if (2*col+1, 2*row+1) in traps_dict and (
                        2*col + 2,
                        2*row + 1,
                    ) in junctions_dict:
                        self._arch.addEdge(
                            traps_dict[(2*col+1, 2*row+1)], junctions_dict[(2*col + 2, 2*row+1)]
                        )

        return self._arch, (instructions, barriers)


import logging
from multiprocessing import get_logger

def process_circuit(distance, capacity, gate_improvements, num_shots):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("process_log.txt")
    formatter = logging.Formatter('%(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting circuit generation for distance {distance} and capacity {capacity}")
  
    circuit = QCCDCircuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
    )
    nqubitsNeeded = 2 * distance**2 - 1
    # ntrapsNeeded = int(nqubitsNeeded / capacity) + 1
    # nrowsNeeded=0
    # while ((nrowsNeeded-1) * (2*nrowsNeeded-1)+nrowsNeeded) < len(ntrapsNeeded):
    #     nrowsNeeded+=1
    # nrowsNeeded+=2
    nrowsNeeded = distance+2

    logger.info(f"Processing circuit with {nqubitsNeeded} qubits and {nrowsNeeded} rows")

    arch, (instructions, _) = circuit.processCircuitAugmentedGrid(rows=nrowsNeeded, cols=nrowsNeeded, trapCapacity=capacity)
    arch.refreshGraph()

    results = {"ElapsedTime": {}, "Operations": {}, "MeanConcurrency": {}, "QubitOperations": {}, "LogicalErrorRates": {}, "PhysicalZErrorRates": {}, "PhysicalXErrorRates": {}}

    for method_, label in zip([arch.processOperationsWithSafety, arch.processOperationsViaRouting][:1], ["Forwarding", "Routing"][:1]):
        logger.info(f"Processing operations using {label} for distance {distance} and capacity {capacity}")
        
        allOps, barriers = method_(instructions, capacity)
        parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
        logicalErrors = []
        physicalZErrors = []
        physicalXErrors = []
        
        for gate_improvement in gate_improvements:
            logicalError, physicalXError, physicalZError = circuit.simulate(allOps, num_shots=num_shots, error_scaling=gate_improvement)
            logicalErrors.append(logicalError)
            physicalZErrors.append(physicalZError)
            physicalXErrors.append(physicalXError)

        logger.info(f"Simulated {label} method with gate improvements for distance {distance}, capacity {capacity}")
        
        
        for op in parallelOpsMap.values():
            op.calculateOperationTime()
            op.calculateFidelity()

        circuit.resetArch()
        arch.refreshGraph()

        results["Capacity"] = capacity
        results["Distance"] = distance
        results["ElapsedTime"][label] = max(parallelOpsMap.keys())
        results["Operations"][label] = len(allOps)
        results["MeanConcurrency"][label] = np.mean([len(op.operations) for op in parallelOpsMap.values()])
        results["QubitOperations"][label] = len(instructions)
        results["LogicalErrorRates"][label] = logicalErrors
        results["PhysicalZErrorRates"][label] = physicalZErrors
        results["PhysicalXErrorRates"][label] = physicalXErrors
        logger.info(f"{distance} {capacity} {label} = {results}")
    
    logger.info(f"Finished processing for distance {distance} and capacity {capacity}")
    return results