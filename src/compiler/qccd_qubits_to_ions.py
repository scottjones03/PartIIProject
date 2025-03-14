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

MAX_ITER = 20_000
        
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



def _minWeightPerfectMatch(A, BSubset, centralizerMatrix, dividerMatrix, nearestCoordsA, nearestDistsA) -> Tuple[float, Sequence[int]]:
    RelBSubset = np.divide((BSubset-centralizerMatrix), dividerMatrix)
    try:
        diffs = np.linalg.norm(
            nearestCoordsA[:, :, None, :] - RelBSubset[None, None, :, :], axis=3
        ) - nearestDistsA[:, :, None]  
        variance_matrix = np.mean(diffs**2, axis=1)
        cost_matrix =distance_matrix(A, RelBSubset)+variance_matrix
        # the Hungarian algorithm 
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        cost_matrix =distance_matrix(A, RelBSubset)
        # the Hungarian algorithm 
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return total_cost, col_ind

def _arrangeClusters(
    clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
    allGridPos: Sequence[Tuple[int, int]],
    nearestNeighbourCount: int = 4,
    biasY: int =1
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
    distmatrixA = distance_matrix(A, A) 
    np.fill_diagonal(distmatrixA, np.inf)  
    nearestindicesA = np.argsort(distmatrixA, axis=1)[:, :nearestNeighbourCount]
    nearestCoordsA = A[nearestindicesA]  
    nearestDistsA = distmatrixA[np.arange(len(A))[:, None], nearestindicesA] 
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
            dis = max(((p[0]-centroidB[0])**2)*biasY,((p[1]-centroidB[1])**2))
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

        if True:

            bottom_edge = sorted([p for p in nextWindow if p[1] == minY], key=lambda p: p[0])
            right_edge = sorted([p for p in nextWindow if p[0] == maxX], key=lambda p: p[1])
            top_edge = sorted([p for p in nextWindow if p[1] == maxY], key=lambda p: p[0], reverse=True)
            left_edge = sorted([p for p in nextWindow if p[0] == minX], key=lambda p: p[1], reverse=True)

            # Concatenate points to form the path around the rectangle
            sortedNextWindow = bottom_edge[:-1] + right_edge[:-1] + top_edge[:-1] + left_edge[:-1]  
            if len(sortedNextWindow)==0:
                sortedNextWindow = bottom_edge
            # assume regularity in code topology boundary conditions
            regSpacing = int(len(sortedNextWindow) / (cardinalityA-len(gauranteedInBSubset)))
            if regSpacing==0:
                break

            InBSubset = [sortedNextWindow[i*regSpacing] for i in range((cardinalityA-len(gauranteedInBSubset))) ]
            cardinalityInBSubset = len(InBSubset)

            BSubset = np.array(InBSubset+gauranteedInBSubset)
            NotInBSubset = [w for w in sortedNextWindow if w not in InBSubset]
            total_cost, _ = _minWeightPerfectMatch(A, BSubset, centralizerMatrix, dividerMatrix, nearestCoordsA, nearestDistsA)
            currentScore=total_cost


            _i = 0
            while True:
                nextInBSubset = []
                nextScore =  np.inf
                for i in range(cardinalityInBSubset):
                    for bnotinBSubset in NotInBSubset:
                        BSubset = np.array(InBSubset[:i]+[bnotinBSubset]+InBSubset[i+1:]+gauranteedInBSubset)

                        total_cost, _ = _minWeightPerfectMatch(A, BSubset, centralizerMatrix, dividerMatrix, nearestCoordsA, nearestDistsA)
                        if total_cost<nextScore:
                            nextInBSubset = InBSubset[:i]+[bnotinBSubset]+InBSubset[i+1:]
                            nextScore = total_cost
                        if total_cost==0:
                            break

                if currentScore <= nextScore:
                    break
                if _i > MAX_ITER:
                    break

                InBSubset = nextInBSubset
                NotInBSubset = [w for w in sortedNextWindow if w not in InBSubset]
                currentScore = nextScore
                if currentScore == 0:
                    break
                _i+=1
            
            if currentScore<bestcost:
                BSubset = np.array(InBSubset+gauranteedInBSubset)
                NotInBSubset = [w for w in sortedNextWindow if w not in InBSubset]
                _, col_ind = _minWeightPerfectMatch(A, BSubset, centralizerMatrix, dividerMatrix, nearestCoordsA, nearestDistsA)
                bestmap = [BSubset[idx] for idx in col_ind]
                bestcost = currentScore
            if currentScore==0:
                break

        # if True:   
        #     # do not assume regularity in code topology boundary conditions
        #     for notGauranteedInBSubset in itertools.combinations(nextWindow, cardinalityA-len(gauranteedInBSubset)):
        #         BSubset = np.array(list(notGauranteedInBSubset)+gauranteedInBSubset)
        #         RelBSubset = np.divide((BSubset-centralizerMatrix), dividerMatrix)
        #         cost_matrix = distance_matrix(A, RelBSubset)
        #         # the Hungarian algorithm 
        #         row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #         total_cost = cost_matrix[row_ind, col_ind].sum()
        #         if total_cost<bestcost:
        #             bestmap = [BSubset[idx] for idx in col_ind]
        #             bestcost = total_cost
    return bestmap, bestcost


def arrangeClusters(
    clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
    allGridPos: Sequence[Tuple[int, int]],
    nearestNeighbourCount: int = 4,
    biasY: int =1
):
    return _arrangeClusters(clusters, allGridPos, nearestNeighbourCount, biasY)[0]

def hillClimbOnArrangeClusters(
    clusters: Sequence[Tuple[Sequence[Ion], Tuple[float, float]]],
    allGridPos: Sequence[Tuple[int, int]],
    nearestNeighbourCount: int = 4
):
    biasY = 1
    currentcost = np.inf
    currentmap = []
    nextcost = np.inf
    nextmap = []

    iters_ = 0
    while True:
        currentmap, currentcost = nextmap, nextcost
        nextcost = np.inf
        nextmap = []
        besti = 0
        for i in range(10):
            _map, _cost = _arrangeClusters(clusters, allGridPos, nearestNeighbourCount, biasY+i)
            if _cost < nextcost:
                besti = i 
                nextcost = _cost 
                nextmap = _map 
            if nextcost == 0:
                break
        if besti < 10 and nextcost<currentcost:
            return nextmap
        if nextcost>=currentcost:
            return currentmap
        if iters_ > MAX_ITER:
            return currentmap 
        biasY +=10