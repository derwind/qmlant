from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def ZFeatureMap(
    feature_dimension: int,
    reps: int = 2,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str = "ZFeatureMap",
    parameter_multiplier = 2.0,
) -> QuantumCircuit:
    qc = QuantumCircuit(feature_dimension)
    feature_map = QuantumCircuit(feature_dimension, name=name)
    pvecs = ParameterVector(parameter_prefix, feature_dimension * reps)

    for rep in range(reps):
        feature_map.h(feature_map.qregs[0][:])
        if insert_barriers:
            feature_map.barrier()
        for i in range(feature_dimension):
            feature_map.rz(parameter_multiplier * pvecs[i], i)
        if insert_barriers and rep + 1 < reps:
            feature_map.barrier()

    return qc.compose(feature_map.to_instruction())
