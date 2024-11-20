import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from typing import (
    Sequence,
    Tuple,
)
import itertools
from src.utils.qccd_nodes import *
from src.utils.qccd_operations import *
from src.utils.qccd_operations_on_qubits import *
from src.utils.qccd_arch import *
from src.compiler.qccd_parallelisation import *

        
def _partitionClusterIons(
    ions: Sequence[Ion], coords: npt.NDArray[np.float_], trapCapacity: int
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

def regularPartition(measurementIons: Sequence[Ion], dataIons: Sequence[Ion], trapCapacity: int):
        dIonsPerTrap = trapCapacity 
        while True:
            measurementIonsL = list(measurementIons)
            measurementIonCoords = np.array([list(ion.pos) for ion in measurementIonsL])
            dataIonsL = list(dataIons)
            dataIonCoords = np.array([list(ion.pos) for ion in dataIonsL])
            clustersD=list(_partitionClusterIons(dataIonsL, dataIonCoords, dIonsPerTrap)) 
            clustersM=list(_partitionClusterIons(measurementIonsL, measurementIonCoords, 1)) 
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
                    ions = list(measurementIons)+list(dataIons)
                    ionCoords = np.array([list(ion.pos) for ion in ions])
                    clusters=_partitionClusterIons(ions, ionCoords, trapCapacity-1)
                    return clusters 
                dIonsPerTrap -= 1
            else:
                return clusters

def arrangeClusters(
    clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
    allGridPos: Sequence[Tuple[int, int]]
):
    A = np.array([c[1] for c in clusters])
    minX, minY = min([w[0] for w in A]), min([w[1] for w in A])
    maxX, maxY = max([w[0] for w in A]), max([w[1] for w in A])
    dX, dY = maxX-minX, maxY-minY
    if dX==0:
        dX = 1
    if dY==0:
        dY = 1
    centralizerMatrix = np.array([(minX, minY) for _ in range(A.shape[0])])
    dividerMatrix = np.array([(dX, dY) for _ in range(A.shape[0])])
    A = np.divide((A-centralizerMatrix), dividerMatrix)
    if len(allGridPos)==0 or len(A)==0:
        return []
    centroidB = np.mean(allGridPos, axis=0)
    sortedToCentroidB = sorted(allGridPos, key=lambda p: (p[0]-centroidB[0])**2+(p[1]-centroidB[1])**2)
    aroundCentroid = []
    for xsign, ysign in [(-1,0), (1,0),(0,1),(0,-1), (0,0),(1,1), (-1,-1), (1,-1),(-1,1)]:
        for p in sortedToCentroidB:
            if xsign == -1 and p[0]<centroidB[0]:
                if ysign == -1 and p[1]<centroidB[1]:
                    aroundCentroid.append(p)
                    break
                elif ysign == 1 and p[1]>centroidB[1]:
                    aroundCentroid.append(p)
                    break
                elif ysign==0:
                    aroundCentroid.append(p)
                    break
            elif xsign == 1 and p[0]>centroidB[0]:
                if ysign == -1 and p[1]<centroidB[1]:
                    aroundCentroid.append(p)
                    break
                elif ysign == 1 and p[1]>centroidB[1]:
                    aroundCentroid.append(p)
                    break
                elif ysign==0:
                    aroundCentroid.append(p)
                    break
            elif xsign==0:
                if ysign == -1 and p[1]<centroidB[1]:
                    aroundCentroid.append(p)
                    break
                elif ysign == 1 and p[1]>centroidB[1]:
                    aroundCentroid.append(p)
                    break

                elif ysign==0:
                    aroundCentroid.append(p)
                    break
    bestcost = np.inf
    bestmap = []
    aroundCentroid = set(aroundCentroid)
    for centroidB in aroundCentroid:
        notPickedYet: Dict[float, List[Tuple[int, int]]] = {}
        for p in allGridPos:
            dis = max((p[0]-centroidB[0])**2,(p[1]-centroidB[1])**2)
            if dis not in notPickedYet:
                notPickedYet[dis]=[p]
            else:
                notPickedYet[dis].append(p)
        cardinalityA = len(A)
        if cardinalityA > len(allGridPos):
            raise ValueError("Not enough traps")
        gauranteedInBSubset = []
        nextWindow = []
        while True:
            nextWindow = notPickedYet.pop(min(notPickedYet.keys()))
            if len(gauranteedInBSubset)+len(nextWindow)<cardinalityA:
                gauranteedInBSubset.extend(nextWindow)
            else:
                break

        minX, minY = min([w[0] for w in nextWindow]), min([w[1] for w in nextWindow])
        maxX, maxY = max([w[0] for w in nextWindow]), max([w[1] for w in nextWindow])
        dX, dY = maxX-minX, maxY-minY
        if dX==0:
            dX = 1
        if dY==0:
            dY = 1
        centralizerMatrix = np.array([(minX, minY) for _ in range(cardinalityA)])
        dividerMatrix = np.array([(dX, dY) for _ in range(cardinalityA)])
        for notGauranteedInBSubset in itertools.combinations(nextWindow, cardinalityA-len(gauranteedInBSubset)):
            BSubset = np.array(list(notGauranteedInBSubset)+gauranteedInBSubset)
            RelBSubset = np.divide((BSubset-centralizerMatrix), dividerMatrix)
            cost_matrix = distance_matrix(A, RelBSubset)
            # the Hungarian algorithm 
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
            if total_cost<bestcost:
                bestmap = [BSubset[idx] for idx in col_ind]
                bestcost = total_cost
    return bestmap