# %%
import stim
import pymatching
import numpy as np
import sinter
from typing import List
import matplotlib.pyplot as plt

# %%
def generate_circuit(rounds=1, before_round_data_depolarization= 0.1, before_measure_flip_probability=0.2, after_reset_flip_probability=0.3, after_clifford_depolarization=0.4):
    return f"""# surface_code circuit exercise.
# rounds: {rounds}
# before_round_data_depolarization: {before_round_data_depolarization}
# before_measure_flip_probability: {before_measure_flip_probability}
# after_reset_flip_probability: {after_reset_flip_probability}
# after_clifford_depolarization: {after_clifford_depolarization}
# layout:
# L00 X01 L02 X03 L04 X05 L06
# Z10 d11 Z12 d13 Z14 d15 Z16
# d20 X21 d22 X22 d24 X25 d26
# Z30 d31 Z32 d33 Z34 d35 Z36
# d40 X41 d42 X42 d44 X45 d46
# Z50 d51 Z52 d53 Z54 d55 Z56
# d60 X61 d62 X62 d64 X65 d66
# Legend:
#     d# = data qubit
#     L# = data qubit with logical observable crossing
#     X# = measurement qubit (X stabilizer)
#     Z# = measurement qubit (Z stabilizer)
QUBIT_COORDS(0, 0) 1
QUBIT_COORDS(0, 1) 2
QUBIT_COORDS(0, 2) 3
QUBIT_COORDS(0, 3) 4
QUBIT_COORDS(0, 4) 5
QUBIT_COORDS(0, 5) 6
QUBIT_COORDS(0, 6) 7 
QUBIT_COORDS(1, 0) 8
QUBIT_COORDS(1, 1) 9
QUBIT_COORDS(1, 2) 10
QUBIT_COORDS(1, 3) 11
QUBIT_COORDS(1, 4) 12
QUBIT_COORDS(1, 5) 13
QUBIT_COORDS(1, 6) 14
QUBIT_COORDS(2, 0) 15
QUBIT_COORDS(2, 1) 16
QUBIT_COORDS(2, 2) 17
QUBIT_COORDS(2, 3) 18
QUBIT_COORDS(2, 4) 19
QUBIT_COORDS(2, 5) 20
QUBIT_COORDS(2, 6) 21
QUBIT_COORDS(3, 0) 22
QUBIT_COORDS(3, 1) 23
QUBIT_COORDS(3, 2) 24
QUBIT_COORDS(3, 3) 25
QUBIT_COORDS(3, 4) 26
QUBIT_COORDS(3, 5) 27
QUBIT_COORDS(3, 6) 28
QUBIT_COORDS(4, 0) 29
QUBIT_COORDS(4, 1) 30
QUBIT_COORDS(4, 2) 31
QUBIT_COORDS(4, 3) 32
QUBIT_COORDS(4, 4) 33
QUBIT_COORDS(4, 5) 34
QUBIT_COORDS(4, 6) 35
QUBIT_COORDS(5, 0) 36
QUBIT_COORDS(5, 1) 37
QUBIT_COORDS(5, 2) 38
QUBIT_COORDS(5, 3) 39
QUBIT_COORDS(5, 4) 40
QUBIT_COORDS(5, 5) 41
QUBIT_COORDS(5, 6) 42
QUBIT_COORDS(6, 0) 43
QUBIT_COORDS(6, 1) 44
QUBIT_COORDS(6, 2) 45
QUBIT_COORDS(6, 3) 46
QUBIT_COORDS(6, 4) 47
QUBIT_COORDS(6, 5) 48
QUBIT_COORDS(6, 6) 49
R 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
X_ERROR({after_reset_flip_probability}) 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
R 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
X_ERROR({after_reset_flip_probability}) 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
TICK
DEPOLARIZE1({before_round_data_depolarization}) 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
TICK
# Plaquette Syndromes 
H 2 4 6 16 18 20 30 32 34 44 46 48
DEPOLARIZE1({after_clifford_depolarization}) 2 4 6 16 18 20 30 32 34 44 46 48
TICK
# Plaquette Syndromes triplets
CX 1 2 3 4 5 6 43 44 45 46 47 48
DEPOLARIZE2({after_clifford_depolarization}) 1 2 3 4 5 6 43 44 45 46 47 48
TICK
CX 9 2 11 4 13 6 37 44 39 46 41 48
DEPOLARIZE2({after_clifford_depolarization}) 9 2 11 4 13 6 37 44 39 46 41 48
TICK
CX 3 2 5 4 7 6 45 44 47 46 49 48
DEPOLARIZE2({after_clifford_depolarization}) 3 2 5 4 7 6 45 44 47 46 49 48
TICK
# Plaquette Syndromes quads
CX 9 16 11 18 13 20 23 30 25 32 27 34
DEPOLARIZE2({after_clifford_depolarization}) 9 16 11 18 13 20 23 30 25 32 27 34
TICK
CX 15 16 17 18 19 20 29 30 31 32 33 34
DEPOLARIZE2({after_clifford_depolarization}) 15 16 17 18 19 20 29 30 31 32 33 34
TICK
CX 17 16 19 18 21 20 31 30 33 32 35 34
DEPOLARIZE2({after_clifford_depolarization}) 17 16 19 18 21 20 31 30 33 32 35 34
TICK
CX 23 16 25 18 27 20 37 30 39 32 41 34
DEPOLARIZE2({after_clifford_depolarization}) 23 16 25 18 34 20 37 30 39 32 41 34
TICK
# Vertex Syndromes triplets
CX 1 8 15 22 29 36 7 14 21 28 35 42
DEPOLARIZE2({after_clifford_depolarization}) 1 8 15 22 29 36 7 14 21 28 35 42
TICK 
CX 15 8 29 22 43 36 21 14 35 28 49 42
DEPOLARIZE2({after_clifford_depolarization}) 1 8 15 22 29 36 7 14 21 28 35 42
TICK 
CX 9 8 23 22 37 36 13 14 27 28 41 42
DEPOLARIZE2({after_clifford_depolarization}) 1 8 15 22 29 36 7 14 21 28 35 42
TICK 
# Vertex Syndromes quads
CX 9 10 11 12 23 24 25 26 37 38 39 40
DEPOLARIZE2({after_clifford_depolarization}) 9 10 11 12 23 24 25 26 37 38 39 40
TICK
CX 3 10 5 12 17 24 19 26 31 38 33 40
DEPOLARIZE2({after_clifford_depolarization}) 3 10 5 12 17 24 19 26 31 38 33 40
TICK
CX 11 10 13 12 25 24 27 26 39 38 41 40
DEPOLARIZE2({after_clifford_depolarization}) 11 10 13 12 25 24 27 26 39 38 41 40
TICK
CX 17 10 19 12 31 24 33 26 45 38 47 40
DEPOLARIZE2({after_clifford_depolarization}) 17 10 19 12 31 24 33 26 45 38 47 40
TICK
# Plaquette Syndromes 
H 2 4 6 16 18 20 30 32 34 44 46 48
DEPOLARIZE1({after_clifford_depolarization}) 2 4 6 16 18 20 30 32 34 44 46 48
TICK
# Measure measurement qubits
X_ERROR({before_measure_flip_probability}) 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
MR 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
X_ERROR({after_reset_flip_probability}) 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
# Detect errors using the vertex Syndromes
DETECTOR(1, 0, 0) rec[-21]
DETECTOR(3, 0, 0) rec[-14]
DETECTOR(5, 0, 0) rec[-7]
DETECTOR(1, 2, 0) rec[-20]
DETECTOR(3, 2, 0) rec[-13]
DETECTOR(5, 2, 0) rec[-6]
DETECTOR(1, 4, 0) rec[-19]
DETECTOR(3, 4, 0) rec[-12]
DETECTOR(3, 4, 0) rec[-5]
DETECTOR(1, 6, 0) rec[-18]
DETECTOR(3, 6, 0) rec[-11]
DETECTOR(5, 6, 0) rec[-4]
REPEAT {rounds}""" + "{" + f"""
    TICK
    DEPOLARIZE1({before_round_data_depolarization}) 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
    TICK
    # Plaquette Syndromes 
    H 2 4 6 16 18 20 30 32 34 44 46 48
    DEPOLARIZE1({after_clifford_depolarization}) 2 4 6 16 18 20 30 32 34 44 46 48
    TICK
    # Plaquette Syndromes triplets
    CX 1 2 3 4 5 6 43 44 45 46 47 48
    DEPOLARIZE2({after_clifford_depolarization}) 1 2 3 4 5 6 43 44 45 46 47 48
    TICK
    CX 9 2 11 4 13 6 37 44 39 46 41 48
    DEPOLARIZE2({after_clifford_depolarization}) 9 2 11 4 13 6 37 44 39 46 41 48
    TICK
    CX 3 2 5 4 7 6 45 44 47 46 49 48
    DEPOLARIZE2({after_clifford_depolarization}) 3 2 5 4 7 6 45 44 47 46 49 48
    TICK
    # Plaquette Syndromes quads
    CX 9 16 11 18 13 20 23 30 25 32 27 34
    DEPOLARIZE2({after_clifford_depolarization}) 9 16 11 18 13 20 23 30 25 32 27 34
    TICK
    CX 15 16 17 18 19 20 29 30 31 32 33 34
    DEPOLARIZE2({after_clifford_depolarization}) 15 16 17 18 19 20 29 30 31 32 33 34
    TICK
    CX 17 16 19 18 21 20 31 30 33 32 35 34
    DEPOLARIZE2({after_clifford_depolarization}) 17 16 19 18 21 20 31 30 33 32 35 34
    TICK
    CX 23 16 25 18 27 20 37 30 39 32 41 34
    DEPOLARIZE2({after_clifford_depolarization}) 23 16 25 18 34 20 37 30 39 32 41 34
    TICK
    # Vertex Syndromes triplets
    CX 1 8 15 22 29 36 7 14 21 28 35 42
    DEPOLARIZE2({after_clifford_depolarization}) 1 8 15 22 29 36 7 14 21 28 35 42
    TICK 
    CX 15 8 29 22 43 36 21 14 35 28 49 42
    DEPOLARIZE2({after_clifford_depolarization}) 1 8 15 22 29 36 7 14 21 28 35 42
    TICK 
    CX 9 8 23 22 37 36 13 14 27 28 41 42
    DEPOLARIZE2({after_clifford_depolarization}) 1 8 15 22 29 36 7 14 21 28 35 42
    TICK 
    # Vertex Syndromes quads
    CX 9 10 11 12 23 24 25 26 37 38 39 40
    DEPOLARIZE2({after_clifford_depolarization}) 9 10 11 12 23 24 25 26 37 38 39 40
    TICK
    CX 3 10 5 12 17 24 19 26 31 38 33 40
    DEPOLARIZE2({after_clifford_depolarization}) 3 10 5 12 17 24 19 26 31 38 33 40
    TICK
    CX 11 10 13 12 25 24 27 26 39 38 41 40
    DEPOLARIZE2({after_clifford_depolarization}) 11 10 13 12 25 24 27 26 39 38 41 40
    TICK
    CX 17 10 19 12 31 24 33 26 45 38 47 40
    DEPOLARIZE2({after_clifford_depolarization}) 17 10 19 12 31 24 33 26 45 38 47 40
    TICK
    # Plaquette Syndromes 
    H 2 4 6 16 18 20 30 32 34 44 46 48
    DEPOLARIZE1({after_clifford_depolarization}) 2 4 6 16 18 20 30 32 34 44 46 48
    TICK
    # Measure measurement qubits
    X_ERROR({before_measure_flip_probability}) 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    MR 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    X_ERROR({after_reset_flip_probability}) 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48
    # Used for adding time?
    SHIFT_COORDS(0, 0, 1)
    # Detecting errors by comparing to previous shot
    DETECTOR(6, 5, 0) rec[-1] rec[-25]
    DETECTOR(6, 3, 0) rec[-2] rec[-26]
    DETECTOR(6, 1, 0) rec[-3] rec[-27]
    DETECTOR(5, 6, 0) rec[-4] rec[-28]
    DETECTOR(5, 4, 0) rec[-5] rec[-29]
    DETECTOR(5, 2, 0) rec[-6] rec[-30]
    DETECTOR(5, 0, 0) rec[-7] rec[-31]
    DETECTOR(4, 5, 0) rec[-8] rec[-32]
    DETECTOR(4, 3, 0) rec[-9] rec[-33]
    DETECTOR(4, 1, 0) rec[-10] rec[-34]
    DETECTOR(3, 6, 0) rec[-11] rec[-35]
    DETECTOR(3, 4, 0) rec[-12] rec[-36]
    DETECTOR(3, 2, 0) rec[-13] rec[-37]
    DETECTOR(3, 0, 0) rec[-14] rec[-38]
    DETECTOR(2, 5, 0) rec[-15] rec[-39]
    DETECTOR(2, 3, 0) rec[-16] rec[-40]
    DETECTOR(2, 1, 0) rec[-17] rec[-41]
    DETECTOR(1, 6, 0) rec[-18] rec[-42]
    DETECTOR(1, 4, 0) rec[-19] rec[-43]
    DETECTOR(1, 2, 0) rec[-20] rec[-44]
    DETECTOR(1, 0, 0) rec[-21] rec[-45]
    DETECTOR(0, 5, 0) rec[-22] rec[-46]
    DETECTOR(0, 3, 0) rec[-23] rec[-47]
    DETECTOR(0, 1, 0) rec[-24] rec[-48]
"""+"}"+f"""
# Final readout of the 25 data qubits in Z basis as we are using vertex syndromes
X_ERROR({before_measure_flip_probability}) 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
M 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49
DETECTOR(1, 0, 1) rec[-18] rec[-21] rec[-25] rec[-46]
DETECTOR(3, 0, 1) rec[-11] rec[-14] rec[-18] rec[-39]
DETECTOR(5, 0, 1) rec[-4] rec[-7] rec[-11] rec[-32]
DETECTOR(1, 2, 1) rec[-17] rec[-20] rec[-21] rec[-24] rec[-45]
DETECTOR(3, 2, 1) rec[-10] rec[-13] rec[-14] rec[-17] rec[-38]
DETECTOR(5, 2, 1) rec[-3] rec[-6] rec[-7] rec[-10] rec[-31]
DETECTOR(1, 4, 1) rec[-16] rec[-19] rec[-20] rec[-23] rec[-44]
DETECTOR(3, 4, 1) rec[-9] rec[-12] rec[-13] rec[-16] rec[-37]
DETECTOR(5, 4, 1) rec[-2] rec[-5] rec[-6] rec[-9] rec[-30]
DETECTOR(1, 6, 1) rec[-15] rec[-19] rec[-22] rec[-43]
DETECTOR(3, 6, 1) rec[-8] rec[-12] rec[-15] rec[-36]
DETECTOR(5, 6, 1) rec[-1] rec[-5] rec[-8] rec[-29]
OBSERVABLE_INCLUDE(0) rec[-22] rec[-23] rec[-24] rec[-25]
"""

# %%
import numpy as np
L = 2

sigma = np.array([[0 for _ in range(L)] for _ in range(L-1)])
for i in range(L-1):
    sigma[i][i] = 1 
    sigma[i][i+1] = 1

idC1 = np.identity(L)
idD1 = np.identity(L-1)
idC0 = np.identity(L-1)
idD0 = np.identity(L)

C1 = np.array([0 for _ in range(L)])
D1 = np.array([0 for _ in range(L-1)])
C0 = np.array([0 for _ in range(L-1)])
D0 = np.array([0 for _ in range(L)])
c1tensord1 = np.kron(C1, D1)
c1tensord0 = np.kron(C1, D0)
c0tensord1 = np.kron(C1, D0)
c0tensord0 = np.kron(C0, D0)
# print(c1tensord1.shape)
# print(c1tensord0.shape)
# print(c0tensord1.shape)
# print(c0tensord0.shape)

m1 = np.kron(sigma, idD1)
m2 = np.kron(idC1, sigma.T)

# print(m1.shape)
# print(m2.shape)

sigma2E = np.concatenate([m1, m2])
sigma1E = np.concatenate([np.kron(idC0, sigma.T).T, np.kron(sigma, idD0).T]).T

# print(sigma2E.shape)
# print(sigma1E.shape)


# print(sigma2E.T)
# print(sigma1E)

dimC0, dimC1, dimD0, dimD1 = L-1,L,L,L-1
dim_C0tD1_p_C1tD0 = dimC0*dimD1 + dimC1*dimD0 
print(dim_C0tD1_p_C1tD0)

E2 = np.ones_like(c1tensord1)
E1 = np.matmul(sigma2E, E2)
E0 = np.matmul(sigma1E, E1)


idE2 = np.identity(E2.shape[0])
idE1 = np.identity(E1.shape[0])
idE0 = np.identity(E0.shape[0])


# print(np.matmul(sigma2E, np.ones_like(idE2)))

e2tensord1 = np.kron(E2, D1)
e0tensord0 = np.kron(E0, D0)
# print(e2tensord1.shape)
# print(e0tensord0.shape)


sigma3F = np.concatenate([np.kron(sigma2E, idD1), np.kron(idE2, sigma.T)])


# print(sigma3F.shape)

sigma2F = np.concatenate([
    np.concatenate([np.kron(sigma1E, idD1).T, np.zeros((np.kron(sigma1E, idD1).shape[0], np.kron(sigma2E, idD0).shape[1])).T]).T,
    np.concatenate([np.kron(idE1, sigma.T).T, np.kron(sigma2E, idD0).T]).T
])
sigma1F = np.concatenate([np.kron(idE0, sigma.T).T, np.kron(sigma1E, idD0).T]).T

# print(sigma2F.shape)
# print(sigma1F.shape)


F3 = e2tensord1 
F2 = np.matmul(sigma3F, F3)
F1 = np.matmul(sigma2F, F2)
F0 = np.matmul(sigma1F, F1)

idF3 = np.identity(F3.shape[0])
idF2 = np.identity(F2.shape[0])
idF1 = np.identity(F1.shape[0])
idF0 = np.identity(F0.shape[0])

sigma4G = np.concatenate([np.kron(sigma3F, idC1), np.kron(idF3, sigma)])
sigma3G = np.concatenate([
    np.concatenate([np.kron(sigma2F, idC1).T, np.zeros((np.kron(sigma2F, idC1).shape[0], np.kron(sigma3F, idC0).shape[1])).T]).T,
    np.concatenate([np.kron(idF2, sigma).T, np.kron(sigma3F, idC0).T]).T
])
sigma2G = np.concatenate([
    np.concatenate([np.kron(sigma1F, idC1).T, np.zeros((np.kron(sigma1F, idC1).shape[0], np.kron(sigma2F, idC0).shape[1])).T]).T,
    np.concatenate([np.kron(idF1, sigma).T, np.kron(sigma2F, idC0).T]).T
])

sigma1G = np.concatenate([np.kron(idF0, sigma).T, np.kron(sigma1F, idC0).T]).T

# print(sigma4G.shape)
# print(sigma3G.shape)
# print(sigma2G.shape)
# print(sigma1G.shape)


G4 = np.ones_like(np.kron(F3, C1))
G3 = np.matmul(sigma4G, G4)
G2 = np.matmul(sigma3G, G3)
G1 = np.matmul(sigma2G, G2)
G0 = np.matmul(sigma1G, G1)

# qubits = G2
# mZ = sigma4G.T
# hZ = sigma3G.T 
# hX = sigma2G
# mX = sigma1G
# print(qubits.shape)
# print(mZ)
# print(hZ)
# print(hX)
# print(mX)


# %%
# 4D
stabilizers = [tuple(st) for st in list(sigma3G.T)+list(sigma2G)]
qubits = len(stabilizers[0])
dataQubitIdxs = [i for i in range(qubits)]
# 4D
stabilizersZ = [tuple(st) for st in list(sigma3G.T)]
stabilizersX = [tuple(st) for st in list(sigma2G)]
metachecksZ = [tuple(st) for st in list(sigma4G.T)]
metachecksX = [tuple(st) for st in list(sigma1G)]


# %%
# 3D
stabilizers = [tuple(st) for st in list(sigma2F.T)+list(sigma1F)]
qubits = len(stabilizers[0])
dataQubitIdxs = [i for i in range(qubits)]
# 3D
stabilizersX = [tuple(st) for st in list(sigma1F)]
stabilizersZ = [tuple(st) for st in list(sigma2F.T)]
metachecksX = []
metachecksZ = [tuple(st) for st in list(sigma3F.T)]


# %%
DIMENSIONS = 2

LGrid = 10



coordsFree = []
coordsToVisit = [tuple([0 for _ in range(DIMENSIONS)])]
while len(coordsToVisit)>0:
    coord = coordsToVisit.pop(0)
    coordsFree.append(coord)
    for j in range(DIMENSIONS):
        ncoord = tuple([(coord[i]+(i==j))%LGrid for i in range(DIMENSIONS)])
        if ncoord not in (coordsToVisit+coordsFree):
            coordsToVisit.append(ncoord)

stabilizersLeft = list(stabilizersX).copy()
stabilizersZLeft = list(stabilizersZ).copy()
currentStabl=max(stabilizersLeft, key=lambda st: sum(st))
stabilizersLeft.remove(currentStabl)
currentZStabl = min(stabilizersZLeft, key=lambda st: sum((v1-v2)**2 for v1, v2 in zip(currentStabl, st)))
stabilizersZLeft.remove(currentZStabl)
currentcoords=tuple([int(LGrid/2) for _ in range(DIMENSIONS)])
stablizersToCoords = {currentStabl: currentcoords}
stablizersToCoords[currentZStabl] = currentcoords
coordsFree.remove(currentcoords)
zStablToXStabl = {currentZStabl: currentStabl}

while len(stabilizersLeft)>0:
    stabl = max(stabilizersLeft, key= lambda st: sum(st[q]==currentStabl[q]==1 for q in range(qubits)))
    stabilizersLeft.remove(stabl)
    if len(coordsFree)==0:
        raise Exception("not enough coords")
    takencoords = tuple([np.mean([c[i] for c in stablizersToCoords.values()]) for i in range(DIMENSIONS)])
    nearestcoord = min(coordsFree, key=lambda c: sum((c[i]-currentcoords[i])**2 for i in range(DIMENSIONS))+sum((c[i]-takencoords[i])**2 for i in range(DIMENSIONS))/(DIMENSIONS*LGrid**2))
    coordsFree.remove(nearestcoord)
    currentcoords = nearestcoord
    currentStabl=stabl
    currentZStabl = min(stabilizersZLeft, key=lambda st: sum((v1-v2)**2 for v1, v2 in zip(currentStabl, st)))
    stabilizersZLeft.remove(currentZStabl)
    stablizersToCoords[currentStabl]=currentcoords
    stablizersToCoords[currentZStabl]=currentcoords
    zStablToXStabl[currentZStabl]=currentStabl


# print(list(stablizersToCoords.values()))

qubitsToManyCoords = {q: [c for st,c in stablizersToCoords.items() if st[q]==1] for q in range(qubits)}
qubitsToCoords = {}
for q, coords in qubitsToManyCoords.items():
    qubitsToCoords[q] = tuple([round(np.mean([c[i] for c in coords]),1)*10 for i in range(DIMENSIONS)])

stabilizersToAncilla = {}
for i, stz in enumerate(stabilizersZ):
    coords = stablizersToCoords[stz]
    qubitsToCoords[i+qubits] = tuple([round(coords[i] ,1)*10 for i in range(DIMENSIONS)])
    stabilizersToAncilla[stz] = i+qubits
    stabilizersToAncilla[zStablToXStabl[stz]] = i+qubits



coordsSet = set(qubitsToCoords.values())
while len(coordsSet) != len(qubitsToCoords):
    doubledCoord = [c for c in coordsSet if sum(qc==c for qc in qubitsToCoords.values())>1][0]
    for q, c in qubitsToCoords.items():
        if c==doubledCoord:
            qubitsToCoords[q] = tuple([c[i]+np.random.randint(2) for i in range(DIMENSIONS)])
            break
    coordsSet = set(qubitsToCoords.values())



# %%
print(len(qubitsToCoords))

# %%
stLXs = stabilizersX[0]
for st in stabilizersX[1:]:
    if all(v1!=v2 for v1, v2 in zip(stLXs, st) if (v1 or v2)):
        stLXs = tuple([(v1 or v2) for v1, v2 in zip(stLXs, st)])
logicalX = [int(stLXs[i]!=1) for i in range(qubits)]


stLZs = stabilizersZ[0]
for st in stabilizersZ[1:]:
    if all(v1!=v2 for v1, v2 in zip(stLZs, st) if (v1 or v2)):
        stLZs = tuple([(v1 or v2) for v1, v2 in zip(stLZs, st)])
logicalZ = [int(stLZs[i]!=1) for i in range(qubits)]


# %%
import stim
qcoords = ""
for q, c in qubitsToCoords.items():
    qcoords+=f"QUBIT_COORDS({int(c[0])}, {int(c[1])}) {q}\n"

circuit = stim.Circuit(qcoords)
circuit.append("R", [q for q in qubitsToCoords.keys()])
circuit.append("TICK")
# Syndromes 
for stx in stabilizersX:
    dataqubits = [i for i, v in enumerate(stx) if v==1]
    ancillaqubit = stabilizersToAncilla[stx]
    for d in dataqubits:
        circuit.append("CNOT",  [d, ancillaqubit])
        circuit.append("TICK")


ndataqubits = qubits
nancillas = len(stabilizersX)

qubitToRec = {}
detectors = []
for i, ancilla in enumerate(set(stabilizersToAncilla.values())):
    circuit.append("MR", [ancilla])
    acoords = qubitsToCoords[ancilla]
    qubitToRec[ancilla] = i-(ndataqubits+2*nancillas)
    detectors.append(stim.Circuit(f"DETECTOR({acoords[0]}, {acoords[1]}, 0) rec[{i-nancillas}]"))
for d in detectors:
    circuit=circuit+d
detector="SHIFT_COORDS(0, 0, 1)"
circuit=circuit+stim.Circuit(detector)
circuit.append("TICK")
circuit.append("H",  [stabilizersToAncilla[stz] for stz in stabilizersZ])
circuit.append("TICK")

for stz in stabilizersZ:
    dataqubits = [i for i, v in enumerate(stz) if v==1]
    ancillaqubit = stabilizersToAncilla[stz]
    for d in dataqubits:
        circuit.append("CNOT",  [d, ancillaqubit])
        circuit.append("TICK")

circuit.append("H",  [stabilizersToAncilla[stz] for stz in stabilizersZ])
circuit.append("TICK")
for i, ancilla in enumerate(set(stabilizersToAncilla.values())):
    circuit.append("M", [ancilla])
circuit.append("TICK")

for q in range(qubits):
    circuit.append("M", [q])
    qubitToRec[q] = q-ndataqubits

for metachk in metachecksX:
    detector=""
    sts = [i for i, v in enumerate(metachk) if v==1]
    ancilla = stabilizersToAncilla[stabilizersX[sts[0]]]
    acoords = qubitsToCoords[ancilla]
    detector+=f"DETECTOR({acoords[0]}, {acoords[1]}, 0) "
    for st in sts:
        anc = stabilizersToAncilla[stabilizersX[st]]
        detector+=f"rec[{qubitToRec[anc]}] "
    detector+="\n"
    circuit=circuit+stim.Circuit(detector)
detector="SHIFT_COORDS(0, 0, 1)"
circuit=circuit+stim.Circuit(detector)


for stx in stabilizersX:
    dataqubits = [i for i, v in enumerate(stx) if v==1]
    ancillaqubit = stabilizersToAncilla[stx]
    acoords = qubitsToCoords[ancillaqubit]
    detector=f"DETECTOR({acoords[0]}, {acoords[1]}, 0) rec[{qubitToRec[ancillaqubit]}] "
    for q in dataqubits:
        detector+=f"rec[{qubitToRec[q]}] "
    detector+="\n"
    circuit=circuit+stim.Circuit(detector)
observable="OBSERVABLE_INCLUDE(0) "
for i, v in enumerate(logicalX):
    if v==1:
        observable+=f"rec[{qubitToRec[i]}] "
observable+="\n"

circuit = circuit+stim.Circuit(observable)



# %%

circuit.without_noise().diagram('timeslice-svg')

# %%

circuitString = str(circuit)
circuitString


# %%
from src.simulator.qccd_circuit import *

nrows = int(np.sqrt(len(qubitsToCoords))+3)
d=4
# safe to have either barriers or go back
barrierThreshold = np.inf
goBackThreshold = 0
for trapCapacity in [2,3,4,5,10,20,30,40,45,50,53,55,60,70,80]:
    noise=0.01
    circuit = QCCDCircuit(circuitString)
    # arch, (instructions, opBarriers) = circuit.processCircuitAugmentedGrid(rows=nrows**2, cols=1, trapCapacity=trapCapacity, padding=(trapCapacity==2), dataQubitIdxs=dataQubitIdxs)
    arch, (instructions, opBarriers) = circuit.processCircuitNetworkedGrid(traps=len(qubitsToCoords), trapCapacity=trapCapacity, dataQubitIdxs=dataQubitIdxs)
    oldPositions = {}
    for idx, (ion, pos) in circuit._ionMapping.items():
        oldPositions[idx] = ion.pos
        ion.set(ion.idx, pos[0], pos[1], parent=ion.parent)

    fig,ax =plt.subplots()
    arch1 = arch
    arch1.refreshGraph()
    edgesDups = []
    edgesPos = []
    ionsInvolved = set()
    score = 0
    for op in instructions:
        if not isinstance(op, TwoQubitMSGate):
            continue
        if not ionsInvolved.isdisjoint(op.ions):
            score+=1
            ionsInvolved=set()
        edgesDups.append((op.ions, score))
        edgesPos.append((op.ionsActedIdxs, score))
        ionsInvolved=ionsInvolved.union(op.ions)
    scores = [score for ((ion1,ion2), score) in edgesDups]
    color_from_score = {s: (5+i*i)*0 for i, s in enumerate( sorted(list(set(scores)), reverse=True))}
    arch1._manipulationTraps.append(([(ion1.idx, ion2.idx) for ((ion1,ion2), score) in edgesDups], [color_from_score[score] for ((ion1,ion2), score) in edgesDups]))
    arch1.display(fig, ax, showLabels=False, showEdges=False, show_junction=False)
    fig.set_size_inches(arch.WINDOW_SIZE[0]*0.9, arch.WINDOW_SIZE[1]*0.9)
    arch1._manipulationTraps = arch1._manipulationTraps[:-1]
    for idx, (ion, pos) in circuit._ionMapping.items():
        ion.set(ion.idx, oldPositions[idx][0], oldPositions[idx][1], parent=ion.parent)

    arch.refreshGraph()

    fig,ax =plt.subplots()
    arch.display(fig, ax, title='map complete', showLabels=False)
    fig.set_size_inches(arch.WINDOW_SIZE[0]*1.5, arch.WINDOW_SIZE[1]*1.5)

    opBarriers = opBarriers if trapCapacity<=barrierThreshold else []
    allOps, barriers = arch.processOperationsWithSafety(instructions, trapCapacity)
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    parallelOps = list(dict(parallelOpsMap).values())

    _, physicalXError, physicalZError = circuit.simulate(allOps, decode=False)

    arch = circuit.resetArch()
    arch.refreshGraph()


    print(f"total number of qubit operations: {len(instructions)}")
    print(f"total number of operations: {len(allOps)}")
    print(f"time for operations: {max(parallelOpsMap.keys())}")
    print(f"physical X and Z errors: {round(physicalXError,6)} {round(physicalZError,6)}")

print("NEXT")
from src.simulator.qccd_circuit import *

nrows = int(np.sqrt(len(qubitsToCoords))+3)
d=4
# safe to have either barriers or go back
barrierThreshold = np.inf
goBackThreshold = 0
for trapCapacity in [2,3,4,5,10,20,30,40,45,50,53,55,60,70,80]:
    noise=0.01
    circuit = QCCDCircuit(circuitString)
    arch, (instructions, opBarriers) = circuit.processCircuitAugmentedGrid(rows=nrows**2, cols=1, trapCapacity=trapCapacity, padding=(trapCapacity==2), dataQubitIdxs=dataQubitIdxs)
    # arch, (instructions, opBarriers) = circuit.processCircuitNetworkedGrid(traps=len(qubitsToCoords), trapCapacity=trapCapacity, dataQubitIdxs=dataQubitIdxs)
    oldPositions = {}
    for idx, (ion, pos) in circuit._ionMapping.items():
        oldPositions[idx] = ion.pos
        ion.set(ion.idx, pos[0], pos[1], parent=ion.parent)

    fig,ax =plt.subplots()
    arch1 = arch
    arch1.refreshGraph()
    edgesDups = []
    edgesPos = []
    ionsInvolved = set()
    score = 0
    for op in instructions:
        if not isinstance(op, TwoQubitMSGate):
            continue
        if not ionsInvolved.isdisjoint(op.ions):
            score+=1
            ionsInvolved=set()
        edgesDups.append((op.ions, score))
        edgesPos.append((op.ionsActedIdxs, score))
        ionsInvolved=ionsInvolved.union(op.ions)
    scores = [score for ((ion1,ion2), score) in edgesDups]
    color_from_score = {s: (5+i*i)*0 for i, s in enumerate( sorted(list(set(scores)), reverse=True))}
    arch1._manipulationTraps.append(([(ion1.idx, ion2.idx) for ((ion1,ion2), score) in edgesDups], [color_from_score[score] for ((ion1,ion2), score) in edgesDups]))
    arch1.display(fig, ax, showLabels=False, showEdges=False, show_junction=False)
    fig.set_size_inches(arch.WINDOW_SIZE[0]*0.9, arch.WINDOW_SIZE[1]*0.9)
    arch1._manipulationTraps = arch1._manipulationTraps[:-1]
    for idx, (ion, pos) in circuit._ionMapping.items():
        ion.set(ion.idx, oldPositions[idx][0], oldPositions[idx][1], parent=ion.parent)

    arch.refreshGraph()

    fig,ax =plt.subplots()
    arch.display(fig, ax, title='map complete', showLabels=False)
    fig.set_size_inches(arch.WINDOW_SIZE[0]*1.5, arch.WINDOW_SIZE[1]*1.5)

    opBarriers = opBarriers if trapCapacity<=barrierThreshold else []
    allOps, barriers = arch.processOperationsWithSafety(instructions, trapCapacity)
    parallelOpsMap = paralleliseOperationsWithBarriers(allOps, barriers)
    parallelOps = list(dict(parallelOpsMap).values())

    _, physicalXError, physicalZError = circuit.simulate(allOps, decode=False)

    arch = circuit.resetArch()
    arch.refreshGraph()


    print(f"total number of qubit operations: {len(instructions)}")
    print(f"total number of operations: {len(allOps)}")
    print(f"time for operations: {max(parallelOpsMap.keys())}")
    print(f"physical X and Z errors: {round(physicalXError,6)} {round(physicalZError,6)}")


# %%
s = """total number of qubit operations: 1066
total number of operations: 24519
time for operations: 0.69565
physical X and Z errors: 0.043866 0.066738
total number of qubit operations: 1066
total number of operations: 16499
time for operations: 0.8133349999999994
physical X and Z errors: 0.122653 0.154251
total number of qubit operations: 1066
total number of operations: 9682
time for operations: 0.4452650000000007
physical X and Z errors: 0.069311 0.082367
total number of qubit operations: 1066
total number of operations: 7689
time for operations: 0.34332000000000057
physical X and Z errors: 0.057931 0.067053
total number of qubit operations: 1066
total number of operations: 3529
time for operations: 0.13244499999999995
physical X and Z errors: 0.03457 0.036634
total number of qubit operations: 1066
total number of operations: 2361
time for operations: 0.09229499999999988
physical X and Z errors: 0.034864 0.035775
total number of qubit operations: 1066
total number of operations: 1477
time for operations: 0.06267499999999998
physical X and Z errors: 0.040025 0.040478
total number of qubit operations: 1066
total number of operations: 1066
time for operations: 0.04292999999999998
physical X and Z errors: 0.056099 0.05597""".splitlines()

qubitOps = []
Ops = []
Times = []
Xerrs = []
Yerrs = []
for l in s:
    if l.startswith('total number of qubit operations:'):
        qubitOps.append( int(l.split('total number of qubit operations: ')[1]))
    elif l.startswith('total number of operations: '):
        Ops.append(int(l.split('total number of operations: ')[1]))
    elif l.startswith('time for operations: '):
        Times.append(float(l.split('time for operations: ')[1]))
    elif l.startswith('physical X and Z errors: '):
        l2 = l.split('physical X and Z errors: ')[1]
        Xerrs.append(float(l2.split(' ')[0]))
        Yerrs.append(float(l2.split(' ')[1]))

cs = [2,3,4,5,10,20,40,80]
_4D_Distance_4_Linear_PhysicalYError = {}
_4D_Distance_4_Linear_PhysicalXError = {}
_4D_Distance_4_Linear_Ops = {}
_4D_Distance_4_Linear_QubitOps = {}
_4D_Distance_4_Linear_ElapsedTime = {}
for i, c in enumerate(cs):
    _4D_Distance_4_Linear_PhysicalXError[c]=[Xerrs[i]]
    _4D_Distance_4_Linear_PhysicalYError[c]=[Yerrs[i]]
    _4D_Distance_4_Linear_Ops[c]=Ops[i]
    _4D_Distance_4_Linear_QubitOps[c]=qubitOps[i]
    _4D_Distance_4_Linear_ElapsedTime[c]=Times[i]

# %%
_4D_Distance_4_Switch_Matrix_PhysicalZErrorRates = {2: [0.008816], 3: [0.016005], 4: [0.01628], 5: [0.017473], 10: [0.022728], 20: [0.028908], 40: [0.041163], 80: [0.05597]}
_4D_Distance_4_Switch_Matrix_PhysicalXErrorRates = {2: [0.007309], 3: [0.013413], 4: [0.014491], 5: [0.015531], 10: [0.021646], 20: [0.028386], 40: [0.040523], 80: [0.056099]}
_4D_Distance_4_Switch_Matrix_Operations = {2: 4090,3: 3922,4: 3444,5: 3340,10: 2711,20: 2182,40: 1669, 80:1066}
_4D_Distance_4_Switch_Matrix_QubitOperations = {2: 1066,3: 1066,4: 1066,5: 1066,10: 1066,20: 1066,40: 1066, 80:1066}
_4D_Distance_4_Switch_Matrix_ElapsedTime = {2: 0.019165000000000005,3: 0.02918500000000001,4: 0.038575000000000005,5:  0.04639999999999999,10: 0.06682499999999994,20: 0.0776449999999999,40: 0.07235999999999992, 80: 0.04292999999999998}

_4D_Distance_4_Augment_Grid_PhysicalZErrorRates = {2: [0.06221], 3: [0.041271], 4: [0.033961], 5: [0.028311], 10: [0.0264], 20: [0.0317], 40: [0.042071], 80: [0.05597]}
_4D_Distance_4_Augment_Grid_PhysicalXErrorRates = {2: [0.046121], 3: [0.033225 ], 4: [0.029133], 5: [0.024859], 10: [0.024636], 20: [0.030855], 40: [0.04132], 80: [ 0.056099]}
_4D_Distance_4_Augment_Grid_Operations = {2: 8460, 3: 5784, 4: 4673, 5: 3856, 10: 2606, 20: 2066, 40: 1861, 80: 1066}
_4D_Distance_4_Augment_Grid_QubitOperations = {2: 1066, 3: 1066, 4: 1066, 5: 1066, 10: 1066, 20: 1066, 40: 1066, 80: 1066}
_4D_Distance_4_Augment_Grid_ElapsedTime = {2: 0.19860999999999984, 3: 0.14558999999999991, 4: 0.12700499999999992, 5: 0.11117499999999991, 10: 0.09665999999999983, 20: 0.08457999999999995, 40: 0.08119499999999989, 80: 0.04292999999999998}


Switch_matrix_Dis_4_surface_code_PhysicalZErrorRates = {2: [0.30511978038651094, 0.15261753138375825, 0.07630876569187912, 0.030523506276751655, 0.015261753138375828, 0.007630876569187914, 0.0015261753138375831, 0.0007630876569187916, 0.0006104701255350327, 0.0005087251046125275, 0.0003815438284593958, 0.00030523506276751634, 0.00021802504483394033, 0.00015261753138375817, 7.630876569187908e-05], 3: [0.5134736226686696, 0.2718270391330609, 0.14270427788329457, 0.057081711153317824, 0.028540855576658912, 0.014270427788329456, 0.002854085557665887, 0.0014270427788329436, 0.0011416342230663565, 0.0009513618525552966, 0.0007135213894164718, 0.0005708171115331782, 0.00040772650823798416, 0.0002854085557665891, 0.00014270427788329456], 4: [0.5259229374764453, 0.28965362716570164, 0.15583887379020264, 0.06233554951608108, 0.03116777475804054, 0.01558388737902027, 0.003116777475804053, 0.0015583887379020266, 0.0012467109903216223, 0.0010389258252680176, 0.0007791943689510133, 0.0006233554951608111, 0.0004452539251148648, 0.00031167774758040557, 0.00015583887379020279], 5: [0.5240544263865105, 0.2898538153328196, 0.15695099833434265, 0.06278039933373708, 0.03139019966686854, 0.01569509983343427, 0.0031390199666868545, 0.0015695099833434273, 0.0012556079866747419, 0.0010463399888956178, 0.0007847549916717136, 0.0006278039933373709, 0.0004484314238124077, 0.00031390199666868546, 0.00015695099833434273], 6: [0.5217188217859168, 0.284711765093645, 0.15422708154638942, 0.061690832618555805, 0.030845416309277902, 0.015422708154638951, 0.00308454163092779, 0.001542270815463895, 0.0012338166523711168, 0.0010281805436425965, 0.0007711354077319475, 0.0006169083261855584, 0.0004406488044182557, 0.0003084541630927792, 0.0001542270815463896], 7: [0.5233004376184166, 0.28666980949786663, 0.15589580597003017, 0.062358322388012095, 0.031179161194006048, 0.015589580597003024, 0.0031179161194006036, 0.0015589580597003018, 0.0012471664477602422, 0.0010393053731335355, 0.0007794790298501509, 0.0006235832238801211, 0.00044541658848580075, 0.00031179161194006054, 0.00015589580597003027], 8: [0.5233004376184166, 0.28666980949786663, 0.15589580597003017, 0.062358322388012095, 0.031179161194006048, 0.015589580597003024, 0.0031179161194006036, 0.0015589580597003018, 0.0012471664477602422, 0.0010393053731335355, 0.0007794790298501509, 0.0006235832238801211, 0.00044541658848580075, 0.00031179161194006054, 0.00015589580597003027], 9: [0.5298563006868999, 0.3198663538363603, 0.16999246810484914, 0.06799698724193963, 0.033998493620969815, 0.016999246810484907, 0.0033998493620969804, 0.0016999246810484902, 0.0013599397448387934, 0.0011332831206989945, 0.0008499623405242451, 0.0006799698724193967, 0.0004856927660138542, 0.00033998493620969835, 0.00016999246810484918], 10: [0.5306322798632986, 0.32330672509706765, 0.17359104025106611, 0.06943654281729297, 0.03471827140864649, 0.017359135704323243, 0.00347182714086465, 0.001735913570432325, 0.0013887308563458602, 0.0011572757136215498, 0.0008679567852161625, 0.0006943654281729301, 0.0004959753058378071, 0.00034718271408646506, 0.00017359135704323253], 11: [0.5281142519873602, 0.32216579473706514, 0.17249719709814146, 0.06900271339159363, 0.03450135669579681, 0.017250678347898406, 0.0034501356695796795, 0.0017250678347898397, 0.0013800542678318716, 0.0011500452231932258, 0.0008625339173949199, 0.0006900271339159358, 0.0004928765242256684, 0.0003450135669579679, 0.00017250678347898395], 12: [0.5281142519873602, 0.32216579473706514, 0.17249719709814146, 0.06900271339159363, 0.03450135669579681, 0.017250678347898406, 0.0034501356695796795, 0.0017250678347898397, 0.0013800542678318716, 0.0011500452231932258, 0.0008625339173949199, 0.0006900271339159358, 0.0004928765242256684, 0.0003450135669579679, 0.00017250678347898395], 13: [0.5281142519873602, 0.32216579473706514, 0.17249719709814146, 0.06900271339159363, 0.03450135669579681, 0.017250678347898406, 0.0034501356695796795, 0.0017250678347898397, 0.0013800542678318716, 0.0011500452231932258, 0.0008625339173949199, 0.0006900271339159358, 0.0004928765242256684, 0.0003450135669579679, 0.00017250678347898395], 14: [0.5281142519873602, 0.32216579473706514, 0.17249719709814146, 0.06900271339159363, 0.03450135669579681, 0.017250678347898406, 0.0034501356695796795, 0.0017250678347898397, 0.0013800542678318716, 0.0011500452231932258, 0.0008625339173949199, 0.0006900271339159358, 0.0004928765242256684, 0.0003450135669579679, 0.00017250678347898395], 15: [0.5281142519873602, 0.32216579473706514, 0.17249719709814146, 0.06900271339159363, 0.03450135669579681, 0.017250678347898406, 0.0034501356695796795, 0.0017250678347898397, 0.0013800542678318716, 0.0011500452231932258, 0.0008625339173949199, 0.0006900271339159358, 0.0004928765242256684, 0.0003450135669579679, 0.00017250678347898395], 30: [0.5067380731077624, 0.43439482632778287, 0.2237616613230777, 0.09142881955716237, 0.04571440977858118, 0.02285720488929059, 0.004571440977858122, 0.002285720488929061, 0.0018285763911432482, 0.0015238136592860385, 0.0011428602444645305, 0.0009142881955716241, 0.0006530629968368745, 0.00045714409778581204, 0.00022857204889290602], 60: [0.4755329470033729, 0.4663379020731151, 0.31316159862597354, 0.12526463945038938, 0.06263231972519469, 0.031316159862597344, 0.006263231972519466, 0.003131615986259733, 0.002505292789007786, 0.0020877439908398225, 0.0015658079931298665, 0.001252646394503893, 0.0008947474246456386, 0.0006263231972519465, 0.00031316159862597323], 120: [0.4755329470033729, 0.4663379020731151, 0.31316159862597354, 0.12526463945038938, 0.06263231972519469, 0.031316159862597344, 0.006263231972519466, 0.003131615986259733, 0.002505292789007786, 0.0020877439908398225, 0.0015658079931298665, 0.001252646394503893, 0.0008947474246456386, 0.0006263231972519465, 0.00031316159862597323]}
Switch_matrix_Dis_4_surface_code_PhysicalXErrorRates = {2: [0.26940401320148555, 0.13470200660074277, 0.06735100330037139, 0.026940401320148553, 0.013470200660074276, 0.006735100330037138, 0.0013470200660074278, 0.0006735100330037139, 0.0005388080264029706, 0.0004490066886691422, 0.00033675501650185695, 0.0002694040132014853, 0.00019243143800106108, 0.00013470200660074266, 6.735100330037133e-05], 3: [0.5026632425078612, 0.28236981734237865, 0.15507139682294657, 0.0620285587291786, 0.0310142793645893, 0.01550713968229465, 0.0031014279364589277, 0.0015507139682294638, 0.0012405711745835722, 0.0010338093121529757, 0.0007753569841147319, 0.0006202855872917861, 0.00044306113377984725, 0.00031014279364589305, 0.00015507139682294652], 4: [0.5158193825014137, 0.29073140746015774, 0.15932794805180775, 0.06373117922072308, 0.03186558961036154, 0.01593279480518077, 0.003186558961036153, 0.0015932794805180765, 0.0012746235844144614, 0.0010621863203453842, 0.0007966397402590382, 0.0006373117922072307, 0.00045522270871945094, 0.00031865589610361535, 0.00015932794805180767], 5: [0.5154589239911118, 0.28772922285219055, 0.1568422596556094, 0.0627369038622437, 0.03136845193112185, 0.015684225965560926, 0.0031368451931121843, 0.0015684225965560922, 0.0012547380772448748, 0.001045615064370728, 0.0007842112982780461, 0.0006273690386224374, 0.0004481207418731695, 0.0003136845193112187, 0.00015684225965560935], 6: [0.5255496252919294, 0.29187392227244774, 0.16067581018865226, 0.06427032407546084, 0.03213516203773042, 0.01606758101886521, 0.003213516203773044, 0.001606758101886522, 0.0012854064815092175, 0.0010711720679243467, 0.000803379050943261, 0.0006427032407546087, 0.00045907374339614927, 0.00032135162037730437, 0.00016067581018865219], 7: [0.5236869855658546, 0.28991208493521237, 0.15924860568993127, 0.06369944227597245, 0.031849721137986224, 0.015924860568993112, 0.003184972113798624, 0.001592486056899312, 0.0012739888455194498, 0.0010616573712662067, 0.000796243028449656, 0.0006369944227597249, 0.0004549960162569464, 0.00031849721137986245, 0.00015924860568993122], 8: [0.5236869855658546, 0.28991208493521237, 0.15924860568993127, 0.06369944227597245, 0.031849721137986224, 0.015924860568993112, 0.003184972113798624, 0.001592486056899312, 0.0012739888455194498, 0.0010616573712662067, 0.000796243028449656, 0.0006369944227597249, 0.0004549960162569464, 0.00031849721137986245, 0.00015924860568993122], 9: [0.5255450620311868, 0.3127993951121326, 0.16611610140703478, 0.0664464405628139, 0.03322322028140695, 0.016611610140703476, 0.003322322028140692, 0.001661161014070346, 0.001328928811256278, 0.0011074406760468967, 0.000830580507035173, 0.000664464405628139, 0.0004746174325915278, 0.0003322322028140695, 0.00016611610140703474], 10: [0.5345182134947762, 0.3229122038445758, 0.17396063502204237, 0.06958665814829743, 0.03479332907414871, 0.017396664537074356, 0.003479332907414872, 0.001739666453707436, 0.0013917331629659491, 0.0011597776358049565, 0.000869833226853718, 0.0006958665814829746, 0.0004970475582021242, 0.0003479332907414873, 0.00017396664537074364], 11: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 12: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 13: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 14: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 15: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 30: [0.5292907801418445, 0.4398663682849053, 0.22969407059737343, 0.09446014304304814, 0.04723007152152407, 0.023615035760762036, 0.004723007152152403, 0.0023615035760762015, 0.001889202860860964, 0.0015743357173841341, 0.0011807517880381008, 0.000944601430430482, 0.0006747153074503439, 0.000472300715215241, 0.0002361503576076205], 60: [0.5097872340425532, 0.48184397163120585, 0.32034730966324965, 0.1281389238652999, 0.06406946193264995, 0.032034730966324976, 0.006406946193264997, 0.0032034730966324985, 0.0025627784773059967, 0.00213564873108833, 0.0016017365483162493, 0.0012813892386529984, 0.0009152780276092855, 0.0006406946193264992, 0.0003203473096632496], 120: [0.5097872340425532, 0.48184397163120585, 0.32034730966324965, 0.1281389238652999, 0.06406946193264995, 0.032034730966324976, 0.006406946193264997, 0.0032034730966324985, 0.0025627784773059967, 0.00213564873108833, 0.0016017365483162493, 0.0012813892386529984, 0.0009152780276092855, 0.0006406946193264992, 0.0003203473096632496]}
Switch_matrix_Dis_4_surface_code_Operations = {2: 1213, 3: 1000, 4: 857, 5: 796, 6: 710, 7: 684, 8: 684, 9: 565, 10: 541, 11: 513, 12: 513, 13: 513, 14: 513, 15: 513, 30: 446, 60: 349, 120: 349}
Switch_matrix_Dis_4_surface_code_ElapsedTime = {2: 0.005325, 3: 0.011870000000000002, 4: 0.01563, 5: 0.016775, 6: 0.017580000000000002, 7: 0.016460000000000002, 8: 0.016460000000000002, 9: 0.019020000000000006, 10: 0.017590000000000005, 11: 0.016575000000000003, 12: 0.016575000000000003, 13: 0.016575000000000003, 14: 0.016575000000000003, 15: 0.016575000000000003, 30: 0.016445000000000005, 60: 0.017689999999999973, 120: 0.017689999999999973}
Switch_matrix_Dis_4_surface_code_QubitOperations = {2: 349, 3: 349, 4: 349, 5: 349, 6: 349, 7: 349, 8: 349, 9: 349, 10: 349, 11: 349, 12: 349, 13: 349, 14: 349, 15: 349, 30: 349, 60: 349, 120: 349}


Grid_Dis_4_surface_code_PhysicalXErrorRates = {2: [0.2698489086937891, 0.13496656103215543, 0.06748328051607771, 0.02699331220643108, 0.01349665610321554, 0.00674832805160777, 0.001349665610321554, 0.000674832805160777, 0.0005398662441286215, 0.0004498885367738511, 0.0003374164025803885, 0.00026993312206431073, 0.00019280937290307921, 0.00013496656103215537, 6.748328051607768e-05], 3: [0.46769829461082246, 0.24626380279939858, 0.12871862455391656, 0.05148744982156665, 0.025743724910783326, 0.012871862455391663, 0.0025743724910783287, 0.0012871862455391644, 0.001029748996431332, 0.0008581241636927775, 0.0006435931227695822, 0.000514874498215666, 0.0003677674987254758, 0.000257437249107833, 0.0001287186245539165], 4: [0.4969048241851098, 0.27154775484540916, 0.1453020429921863, 0.05812081719687448, 0.02906040859843724, 0.01453020429921862, 0.002906040859843723, 0.0014530204299218615, 0.0011624163439374895, 0.0009686802866145746, 0.0007265102149609308, 0.0005812081719687447, 0.00041514869426338935, 0.00029060408598437237, 0.00014530204299218619], 5: [0.5081317924828416, 0.2787348697694036, 0.15002708973215997, 0.0600108358928639, 0.03000541794643195, 0.015002708973215975, 0.003000541794643197, 0.0015002708973215985, 0.0012002167178572795, 0.0010001805982143984, 0.0007501354486607992, 0.0006001083589286398, 0.0004286488278061711, 0.0003000541794643199, 0.00015002708973215994], 6: [0.5129057600967456, 0.27793691539225385, 0.1496609560858078, 0.059864382434323016, 0.029932191217161508, 0.014966095608580754, 0.002993219121716154, 0.001496609560858077, 0.0011972876486864628, 0.0009977397072387178, 0.0007483047804290385, 0.0005986438243432314, 0.0004276027316737365, 0.0002993219121716157, 0.00014966095608580785], 7: [0.5151025311165764, 0.28008664156864027, 0.15135704032388964, 0.06054281612955574, 0.03027140806477787, 0.015135704032388935, 0.0030271408064777893, 0.0015135704032388946, 0.001210856322591117, 0.0010090469354925957, 0.0007567852016194473, 0.0006054281612955585, 0.00043244868663968434, 0.00030271408064777924, 0.00015135704032388962], 8: [0.5151025311165764, 0.28008664156864027, 0.15135704032388964, 0.06054281612955574, 0.03027140806477787, 0.015135704032388935, 0.0030271408064777893, 0.0015135704032388946, 0.001210856322591117, 0.0010090469354925957, 0.0007567852016194473, 0.0006054281612955585, 0.00043244868663968434, 0.00030271408064777924, 0.00015135704032388962], 9: [0.5310638297872345, 0.31618415924878274, 0.16736660341351878, 0.06694664136540746, 0.03347332068270373, 0.016736660341351864, 0.00334733206827037, 0.001673666034135185, 0.0013389328273081503, 0.0011157773560901236, 0.0008368330170675925, 0.0006694664136540751, 0.00047819029546719593, 0.0003347332068270376, 0.0001673666034135188], 10: [0.5346099290780146, 0.32118274224588533, 0.17168240501678486, 0.06867307973656857, 0.034336539868284284, 0.017168269934142142, 0.003433653986828426, 0.001716826993414213, 0.0013734615947313727, 0.0011445513289428087, 0.0008584134967071065, 0.0006867307973656864, 0.0004905219981183467, 0.0003433653986828432, 0.0001716826993414216], 11: [0.5328368794326246, 0.32040592627931547, 0.17080773806574828, 0.06832665782456979, 0.03416332891228489, 0.017081664456142447, 0.003416332891228486, 0.001708166445614243, 0.0013665331564913967, 0.0011387776304094955, 0.0008540832228071215, 0.0006832665782456984, 0.0004880475558897841, 0.0003416332891228492, 0.0001708166445614246], 12: [0.5328368794326246, 0.32040592627931547, 0.17080773806574828, 0.06832665782456979, 0.03416332891228489, 0.017081664456142447, 0.003416332891228486, 0.001708166445614243, 0.0013665331564913967, 0.0011387776304094955, 0.0008540832228071215, 0.0006832665782456984, 0.0004880475558897841, 0.0003416332891228492, 0.0001708166445614246], 13: [0.5328368794326246, 0.32040592627931547, 0.17080773806574828, 0.06832665782456979, 0.03416332891228489, 0.017081664456142447, 0.003416332891228486, 0.001708166445614243, 0.0013665331564913967, 0.0011387776304094955, 0.0008540832228071215, 0.0006832665782456984, 0.0004880475558897841, 0.0003416332891228492, 0.0001708166445614246], 14: [0.5328368794326246, 0.32040592627931547, 0.17080773806574828, 0.06832665782456979, 0.03416332891228489, 0.017081664456142447, 0.003416332891228486, 0.001708166445614243, 0.0013665331564913967, 0.0011387776304094955, 0.0008540832228071215, 0.0006832665782456984, 0.0004880475558897841, 0.0003416332891228492, 0.0001708166445614246], 15: [0.5328368794326246, 0.32040592627931547, 0.17080773806574828, 0.06832665782456979, 0.03416332891228489, 0.017081664456142447, 0.003416332891228486, 0.001708166445614243, 0.0013665331564913967, 0.0011387776304094955, 0.0008540832228071215, 0.0006832665782456984, 0.0004880475558897841, 0.0003416332891228492, 0.0001708166445614246], 30: [0.5221985815602842, 0.4331379846078218, 0.222807355802712, 0.09095157192209741, 0.045475785961048704, 0.022737892980524352, 0.0045475785961048695, 0.0022737892980524347, 0.0018190314384419494, 0.0015158595320349552, 0.0011368946490262174, 0.0009095157192209747, 0.000649654085157839, 0.00045475785961048736, 0.00022737892980524368], 60: [0.5097872340425532, 0.48184397163120585, 0.32034730966324965, 0.1281389238652999, 0.06406946193264995, 0.032034730966324976, 0.006406946193264997, 0.0032034730966324985, 0.0025627784773059967, 0.00213564873108833, 0.0016017365483162493, 0.0012813892386529984, 0.0009152780276092855, 0.0006406946193264992, 0.0003203473096632496], 120: [0.5097872340425532, 0.48184397163120585, 0.32034730966324965, 0.1281389238652999, 0.06406946193264995, 0.032034730966324976, 0.006406946193264997, 0.0032034730966324985, 0.0025627784773059967, 0.00213564873108833, 0.0016017365483162493, 0.0012813892386529984, 0.0009152780276092855, 0.0006406946193264992, 0.0003203473096632496]}
Grid_Dis_4_surface_code_PhysicalZErrorRates = {2: [0.26940401320148555, 0.13470200660074277, 0.06735100330037139, 0.026940401320148553, 0.013470200660074276, 0.006735100330037138, 0.0013470200660074278, 0.0006735100330037139, 0.0005388080264029706, 0.0004490066886691422, 0.00033675501650185695, 0.0002694040132014853, 0.00019243143800106108, 0.00013470200660074266, 6.735100330037133e-05], 3: [0.5026632425078612, 0.28236981734237865, 0.15507139682294657, 0.0620285587291786, 0.0310142793645893, 0.01550713968229465, 0.0031014279364589277, 0.0015507139682294638, 0.0012405711745835722, 0.0010338093121529757, 0.0007753569841147319, 0.0006202855872917861, 0.00044306113377984725, 0.00031014279364589305, 0.00015507139682294652], 4: [0.5158193825014137, 0.29073140746015774, 0.15932794805180775, 0.06373117922072308, 0.03186558961036154, 0.01593279480518077, 0.003186558961036153, 0.0015932794805180765, 0.0012746235844144614, 0.0010621863203453842, 0.0007966397402590382, 0.0006373117922072307, 0.00045522270871945094, 0.00031865589610361535, 0.00015932794805180767], 5: [0.5154589239911118, 0.28772922285219055, 0.1568422596556094, 0.0627369038622437, 0.03136845193112185, 0.015684225965560926, 0.0031368451931121843, 0.0015684225965560922, 0.0012547380772448748, 0.001045615064370728, 0.0007842112982780461, 0.0006273690386224374, 0.0004481207418731695, 0.0003136845193112187, 0.00015684225965560935], 6: [0.5255496252919294, 0.29187392227244774, 0.16067581018865226, 0.06427032407546084, 0.03213516203773042, 0.01606758101886521, 0.003213516203773044, 0.001606758101886522, 0.0012854064815092175, 0.0010711720679243467, 0.000803379050943261, 0.0006427032407546087, 0.00045907374339614927, 0.00032135162037730437, 0.00016067581018865219], 7: [0.5236869855658546, 0.28991208493521237, 0.15924860568993127, 0.06369944227597245, 0.031849721137986224, 0.015924860568993112, 0.003184972113798624, 0.001592486056899312, 0.0012739888455194498, 0.0010616573712662067, 0.000796243028449656, 0.0006369944227597249, 0.0004549960162569464, 0.00031849721137986245, 0.00015924860568993122], 8: [0.5236869855658546, 0.28991208493521237, 0.15924860568993127, 0.06369944227597245, 0.031849721137986224, 0.015924860568993112, 0.003184972113798624, 0.001592486056899312, 0.0012739888455194498, 0.0010616573712662067, 0.000796243028449656, 0.0006369944227597249, 0.0004549960162569464, 0.00031849721137986245, 0.00015924860568993122], 9: [0.5255450620311868, 0.3127993951121326, 0.16611610140703478, 0.0664464405628139, 0.03322322028140695, 0.016611610140703476, 0.003322322028140692, 0.001661161014070346, 0.001328928811256278, 0.0011074406760468967, 0.000830580507035173, 0.000664464405628139, 0.0004746174325915278, 0.0003322322028140695, 0.00016611610140703474], 10: [0.5345182134947762, 0.3229122038445758, 0.17396063502204237, 0.06958665814829743, 0.03479332907414871, 0.017396664537074356, 0.003479332907414872, 0.001739666453707436, 0.0013917331629659491, 0.0011597776358049565, 0.000869833226853718, 0.0006958665814829746, 0.0004970475582021242, 0.0003479332907414873, 0.00017396664537074364], 11: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 12: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 13: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 14: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 15: [0.5381560283687948, 0.3265319064559356, 0.1757743644654502, 0.07035845582522363, 0.03517922791261181, 0.017589613956305906, 0.003517922791261184, 0.001758961395630592, 0.0014071691165044751, 0.0011726409304203935, 0.000879480697815296, 0.0007035845582522376, 0.0005025603987515974, 0.0003517922791261188, 0.0001758961395630594], 30: [0.5292907801418445, 0.4398663682849053, 0.22969407059737343, 0.09446014304304814, 0.04723007152152407, 0.023615035760762036, 0.004723007152152403, 0.0023615035760762015, 0.001889202860860964, 0.0015743357173841341, 0.0011807517880381008, 0.000944601430430482, 0.0006747153074503439, 0.000472300715215241, 0.0002361503576076205], 60: [0.5097872340425532, 0.48184397163120585, 0.32034730966324965, 0.1281389238652999, 0.06406946193264995, 0.032034730966324976, 0.006406946193264997, 0.0032034730966324985, 0.0025627784773059967, 0.00213564873108833, 0.0016017365483162493, 0.0012813892386529984, 0.0009152780276092855, 0.0006406946193264992, 0.0003203473096632496], 120: [0.5097872340425532, 0.48184397163120585, 0.32034730966324965, 0.1281389238652999, 0.06406946193264995, 0.032034730966324976, 0.006406946193264997, 0.0032034730966324985, 0.0025627784773059967, 0.00213564873108833, 0.0016017365483162493, 0.0012813892386529984, 0.0009152780276092855, 0.0006406946193264992, 0.0003203473096632496]}
Grid_Dis_4_surface_code_Operations = {2: 937, 3: 917, 4: 782, 5: 742, 6: 632, 7: 619, 8: 619, 9: 534, 10: 500, 11: 494, 12: 494, 13: 494, 14: 494, 15: 494, 30: 418, 60: 349, 120: 349}
Grid_Dis_4_surface_code_ElapsedTime  = {2: 0.005215000000000001, 3: 0.014295000000000002, 4: 0.015560000000000001, 5: 0.020225000000000007, 6: 0.018045000000000002, 7: 0.017885, 8: 0.017885, 9: 0.015575, 10: 0.014695000000000003, 11: 0.014845000000000002, 12: 0.014845000000000002, 13: 0.014845000000000002, 14: 0.014845000000000002, 15: 0.014845000000000002, 30: 0.015700000000000006, 60: 0.017689999999999973, 120: 0.017689999999999973}
Grid_Dis_4_surface_code_QubitOperations = {2: 349, 3: 349, 4: 349, 5: 349, 6: 349, 7: 349, 8: 349, 9: 349, 10: 349, 11: 349, 12: 349, 13: 349, 14: 349, 15: 349, 30: 349, 60: 349, 120: 349}

# %%
Switch_matrix_data = ...

# %%
cs = list(Switch_matrix_Dis_4_surface_code_Operations.keys())
Switch_matrix__Dis_4_surface_code_PhysicalZErrorRates = {c: Switch_matrix_data['PhysicalZErrorRates']['Forwarding'][4][c] for c in cs}
Switch_matrix__Dis_4_surface_code_PhysicalXErrorRates = {c: Switch_matrix_data['PhysicalXErrorRates']['Forwarding'][4][c] for c in cs}
Switch_matrix__Dis_4_surface_code_Operations = {c: Switch_matrix_data['Operations']['Forwarding'][4][c] for c in cs}
Switch_matrix__Dis_4_surface_code_ElapsedTimes = {c: Switch_matrix_data['ElapsedTime']['Forwarding'][4][c] for c in cs}
Switch_matrix__Dis_4_surface_code_QubitOperations = {c: Switch_matrix_data['QubitOperations']['Forwarding'][4][c] for c in cs}

# %%
plt.rcParams.update({'font.size': 25})

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_size_inches(12,3.5)
_cs = list(_4D_Distance_4_Augment_Grid_ElapsedTime.keys())
# ax.plot(_cs, [_4D_Distance_4_Linear_ElapsedTime[c]*1e3 for c in _cs], label='4D Surface Code on Linear', c='lightblue')
ax.plot(_cs, [_4D_Distance_4_Augment_Grid_ElapsedTime[c]*1e3 for c in _cs], label='4D Surface Code on Grid', c='lightblue')
ax.plot(_cs, [_4D_Distance_4_Switch_Matrix_ElapsedTime[c]*1e3 for c in _cs], label='4D Surface Code on Switch', c='blue')
ax.legend()
ax.set_xlabel('Trap Capacity')
ax.set_xlim(2, 80)
ax.set_ylabel('Elapsed Time (ms)')
ax.set_title('Effect of QEC code on Elapsed Time for Different QCCD Topologies')
_cs = list(Switch_matrix_Dis_4_surface_code_ElapsedTime.keys())
# fig, ax = plt.subplots()
# ax.plot(_cs, [4*Linear__Dis_4_surface_code_ElapsedTimes[c]*1e3 for c in _cs], label='2D Surface Code on Linear', c='orange')
# ax.plot(_cs, [4*Switch_matrix_Dis_4_surface_code_ElapsedTime[c]*1e3 for c in _cs], label='2D Surface Code on Switch', c='red')
ax.plot(_cs, [4*Grid_Dis_4_surface_code_ElapsedTime[c]*1e3 for c in _cs], label='2D Surface Code on Grid',c='red')
ax.legend(prop={'size':22})
ax.set_xlabel('Trap Capacity')
ax.set_xlim(2, 80)
# ax.set_xticks([2,4,8,16,32,64])
# ax.set_xscale('log')

ax.set_ylabel('QEC Cycle Time (ms)')
ax.set_title('Effect of Code Dimension on QEC Cycle Time')

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

_cs = list(_4D_Distance_4_Augment_Grid_Operations.keys())
ax.plot(_cs, [_4D_Distance_4_Linear_Ops[c] for c in _cs], label='Linear Topology', c='blue')
ax.plot(_cs, [_4D_Distance_4_Augment_Grid_Operations[c] for c in _cs], label='Grid Topology', c='green')
ax.plot(_cs, [_4D_Distance_4_Switch_Matrix_Operations[c] for c in _cs], label='Switch Topology', c='red')
# _cs = list(Switch_matrix_Dis_4_surface_code_Operations.keys())
# ax.plot(_cs, [4*Linear__Dis_4_surface_code_Operations[c] for c in _cs], label='Linear Topology', c='blue')
# ax.plot(_cs, [4*Grid_Dis_4_surface_code_Operations[c] for c in _cs], label='Grid Topology', c='green')
# ax.plot(_cs, [4*Switch_matrix_Dis_4_surface_code_Operations[c] for c in _cs], label='Switch Topology', c='darkred')
ax.legend()
ax.set_xlabel('Trap Capacity')
ax.set_xlim(2, 20)
ax.set_ylabel('Total Number of Operations')
ax.set_title('Topology Matters for 4D Surface Code')

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

_cs = list(_4D_Distance_4_Augment_Grid_Operations.keys())
ax.plot(_cs, [_4D_Distance_4_Linear_PhysicalXError[c][0] for c in _cs], label='Linear Topology', c='lightblue')
ax.plot(_cs, [_4D_Distance_4_Augment_Grid_PhysicalXErrorRates[c][0] for c in _cs], label='Grid Topology', c='blue')
ax.plot(_cs, [_4D_Distance_4_Switch_Matrix_PhysicalXErrorRates[c][0] for c in _cs], label='Switch Topology', c='purple')
_cs = list(Switch_matrix_Dis_4_surface_code_Operations.keys())
ax.plot(_cs, [Linear__Dis_4_surface_code_PhysicalXErrorRates[c][5] for c in _cs], label='Linear Topology', c='lightgreen')
ax.plot(_cs, [Grid_Dis_4_surface_code_PhysicalXErrorRates[c][5] for c in _cs], label='Grid Topology', c='green')
ax.plot(_cs, [Switch_matrix_Dis_4_surface_code_PhysicalXErrorRates[c][5] for c in _cs], label='Switch Topology', c='darkgreen')
ax.legend()
ax.set_xlabel('Trap Capacity')
ax.set_xlim(3, 20)
ax.set_ylabel('Mean Physical X Error Rate')
ax.set_title('Topology Matters for 4D Surface Code')

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

_cs = list(_4D_Distance_4_Augment_Grid_Operations.keys())
ax.plot(_cs, [_4D_Distance_4_Augment_Grid_QubitOperations[c] for c in _cs], label='Qubit Ops. for 4D Surface Code', c='lightblue')
ax.plot(_cs, [_4D_Distance_4_Augment_Grid_Operations[c] for c in _cs], label='Total Ops. 4D Surface Code', c='darkblue')
_cs = list(Switch_matrix_Dis_4_surface_code_Operations.keys())
ax.plot(_cs, [4*Grid_Dis_4_surface_code_QubitOperations[c] for c in _cs], label='Qubit Ops. for 2D Surface Code', c='lightgreen')
ax.plot(_cs, [4*Grid_Dis_4_surface_code_Operations[c] for c in _cs], label='Total Ops. for 2D Surface Code', c='darkgreen')
ax.legend()
ax.set_xlabel('Trap Capacity')
ax.set_xlim(2, 80)
ax.set_ylabel('Number of Operations')
ax.set_title('Effect of QEC code on Number of Operations on Grid Topology')


