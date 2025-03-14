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


    @staticmethod
    def _calculateHeatingRate(wiseArch: QCCDWiseArch) -> float:
        globalHeatingRate = 0.0

        globalHeatingRate+=Split.HEATING_RATE/wiseArch.k+Move.HEATING_RATE+JunctionCrossing.HEATING_RATE

        globalHeatingRate+=Merge.HEATING_RATE/wiseArch.k+Move.HEATING_RATE+JunctionCrossing.HEATING_RATE
        return globalHeatingRate
    
    @staticmethod
    def _calculateReconfigTime(wiseArch: QCCDWiseArch) -> float:
        globalReconfigTime = 0.0

        globalReconfigTime+=Split.SPLITTING_TIME+Move.MOVING_TIME+JunctionCrossing.CROSSING_TIME
   
        nrswaps = wiseArch.m
        globalReconfigTime+=PhysicalCrossingSwap.CROSSING_TIME*nrswaps
        for i in range(wiseArch.k):
            ncswaps =wiseArch.n
            globalReconfigTime+=(JunctionCrossing.CROSSING_TIME+Move.MOVING_TIME+PhysicalCrossingSwap.CROSSING_TIME+Move.MOVING_TIME+JunctionCrossing.CROSSING_TIME)*ncswaps
            globalReconfigTime+=Split.SPLITTING_TIME+Move.MOVING_TIME+JunctionCrossing.CROSSING_TIME+PhysicalCrossingSwap.CROSSING_TIME+JunctionCrossing.CROSSING_TIME+Move.MOVING_TIME+Merge.MERGING_TIME #Row swap
            globalReconfigTime+=3*40e-6 #Move Ion to Back of Queue

        nrswaps = wiseArch.m

        globalReconfigTime+=PhysicalCrossingSwap.CROSSING_TIME*nrswaps

        globalReconfigTime+=Merge.MERGING_TIME
        return globalReconfigTime
    
    def calculateOperationTime(self) -> None:
        self._operationTime =  GlobalReconfigurations._calculateReconfigTime(self._wiseArch)

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
        cls, arrangement: Mapping[Trap, Sequence[Ion]], wiseArch: QCCDWiseArch
    ):
        def run():
            reconfigTime = cls._calculateReconfigTime(wiseArch)
            heatingRate = cls._calculateHeatingRate(wiseArch)
            for trap in arrangement.keys():
                while trap.ions:
                    trap.removeIon(trap.ions[0])
            for trap, ions in arrangement.items():
                for i, ion in enumerate(ions):
                    trap.addIon(ion, offset=i)
                    ion.addMotionalEnergy(reconfigTime*heatingRate)
        return cls(
            run=lambda _: run(),
            involvedComponents=list(arrangement.keys()),
            wiseArch=wiseArch
        )


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
    
