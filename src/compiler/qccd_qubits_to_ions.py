import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from typing import (
    Sequence,
    Tuple,
)
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