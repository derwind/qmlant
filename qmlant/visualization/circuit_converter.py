from __future__ import annotations

import math

import quimb as qu
import quimb.tensor as qtn
from qiskit import QuantumCircuit, qasm2
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
    CustomInstruction,
    OpCode,
    bytecode_from_string,
)
from qiskit.circuit import library as lib


def circuit_to_quimb_tn(qc: QuantumCircuit) -> qtn.Circuit | None:
    custom_instructions = tuple(qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    bytecode = bytecode_from_string(
        qc.qasm(),
        [qasm2._normalize_path(path) for path in qasm2.LEGACY_INCLUDE_PATH],
        [
            CustomInstruction(x.name, x.num_params, x.num_qubits, x.builtin)
            for x in custom_instructions
        ],
        tuple(qasm2.LEGACY_CUSTOM_CLASSICAL),
        strict=False,
    )

    gates = []
    has_u, has_cx = False, False
    for custom in custom_instructions:
        gates.append(custom.constructor)
        if custom.name == "U":
            has_u = True
        elif custom.name == "CX":
            has_cx = True
    if not has_u:
        gates.append(lib.UGate)
    if not has_cx:
        gates.append(lib.CXGate)

    circuit: qtn.Circuit | None = None
    num_qubits = 0

    bc = iter(bytecode)
    for op in bc:
        opcode = op.opcode
        if opcode == OpCode.Gate:
            if circuit is None:
                circuit = qtn.Circuit(num_qubits)

            gate_id, parameters, op_qubits = op.operands
            gate = gates[gate_id]

            if gate == lib.standard_gates.HGate:
                circuit.apply_gate("H", *op_qubits)
            elif gate == lib.standard_gates.IGate:
                pass
            elif gate == lib.standard_gates.RXGate:
                circuit.apply_gate("RX", *parameters, *op_qubits)
            elif gate == lib.standard_gates.RYGate:
                circuit.apply_gate("RY", *parameters, *op_qubits)
            elif gate == lib.standard_gates.RZGate:
                circuit.apply_gate("RZ", *parameters, *op_qubits)
            elif gate == lib.standard_gates.SwapGate:
                circuit.apply_gate("SWAP", *parameters, *op_qubits)
            elif gate == lib.standard_gates.PhaseGate:
                circuit.apply_gate(qu.phase_gate(*parameters), *op_qubits)
            elif gate == lib.standard_gates.SGate:
                circuit.apply_gate("S", *op_qubits)
            elif gate == lib.standard_gates.TGate:
                circuit.apply_gate("T", *op_qubits)
            elif gate == lib.standard_gates.TdgGate:
                circuit.apply_gate("U1", -math.pi / 4, *op_qubits)
            elif gate == lib.standard_gates.XGate:
                circuit.apply_gate("X", *op_qubits)
            elif gate == lib.standard_gates.YGate:
                circuit.apply_gate("Y", *op_qubits)
            elif gate == lib.standard_gates.ZGate:
                circuit.apply_gate("Z", *op_qubits)
            elif gate == lib.standard_gates.CXGate:
                circuit.apply_gate("CX", *op_qubits)
            elif gate == lib.standard_gates.CYGate:
                circuit.apply_gate("CY", *op_qubits)
            elif gate == lib.standard_gates.CZGate:
                circuit.apply_gate("CZ", *op_qubits)
            else:
                print(f"Not implemented for a gate: {gate}")
        elif opcode == OpCode.ConditionedGate:
            pass
        elif opcode == OpCode.Measure:
            pass
        elif opcode == OpCode.ConditionedMeasure:
            pass
        elif opcode == OpCode.Reset:
            pass
        elif opcode == OpCode.ConditionedReset:
            pass
        elif opcode == OpCode.Barrier:
            pass
        elif opcode == OpCode.DeclareQreg:
            _, size = op.operands
            num_qubits += size
        elif opcode == OpCode.DeclareCreg:
            pass
        elif opcode == OpCode.SpecialInclude:
            pass
        elif opcode == OpCode.DeclareGate:
            pass
        elif opcode == OpCode.DeclareOpaque:
            pass
        else:
            raise ValueError(f"invalid operation: {op}")

    return circuit
