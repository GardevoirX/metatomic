from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import (
    AtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
    pick_output,
)


try:
    import nvalchemiops.torch.interactions.dispersion._dftd3  # noqa: F401

    HAS_NVALCHEMIOPS = True
except ImportError:
    HAS_NVALCHEMIOPS = False


_REQUIRED_D3_PARAM_KEYS = ("rcov", "r4r2", "c6ab", "cn_ref")
_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _variant_key(name: str, variant: Optional[str]) -> str:
    if variant is None:
        return name
    return f"{name}/{variant}"


def _as_float(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"'{name}' must be a scalar tensor, got {value.shape}")
        return float(value.detach().cpu().item())
    return float(value)


def _torch_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_BY_NAME:
        raise ValueError(f"unsupported model dtype for DFTD3: {name!r}")
    return _DTYPE_BY_NAME[name]


class DFTD3(torch.nn.Module):
    """
    :py:class:`DFTD3` wraps an :py:class:`AtomisticModel` and adds a DFT-D3(BJ)
    dispersion correction computed by ``nvalchemiops``.

    The wrapped energy output is corrected directly. Because the NVIDIA custom op does
    not expose autograd through the corrected energy, corrected forces and stresses are
    exposed as forward outputs under ``non_conservative_forces`` and
    ``non_conservative_stress``. When using ASE, request those through
    ``MetatomicCalculator(..., non_conservative=True)``.

    All D3 tables and damping parameters must already be expressed in the wrapped
    model's length and energy units.
    """

    _requested_neighbor_lists: List[NeighborListOptions]
    _requested_inputs: Dict[str, ModelOutput]

    def __init__(
        self,
        model: AtomisticModel,
        d3_params: Dict[str, torch.Tensor],
        damping_params: Dict[str, float],
        cutoff: float,
        atomic_types_to_numbers: Optional[Dict[int, int]] = None,
        variants: Optional[Dict[str, Optional[str]]] = None,
    ):
        super().__init__()

        if not HAS_NVALCHEMIOPS:
            raise ImportError(
                "DFTD3 requires the optional 'nvalchemi-toolkit-ops[torch]' package"
            )

        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        for key in _REQUIRED_D3_PARAM_KEYS:
            if key not in d3_params:
                raise KeyError(f"missing required D3 parameter table '{key}'")

        for key in ("a1", "a2", "s8"):
            if key not in damping_params:
                raise KeyError(f"missing required D3 damping parameter '{key}'")

        assert isinstance(model, AtomisticModel)
        self._model = model.module
        self._requested_inputs = model.requested_inputs()

        capabilities = model.capabilities()
        outputs = capabilities.outputs.copy()

        has_energy = any(key == "energy" or key.startswith("energy/") for key in outputs)
        if not has_energy:
            raise ValueError(
                "The wrapped model must provide an energy output to use DFTD3."
            )

        if capabilities.length_unit == "":
            raise ValueError("DFTD3 requires the wrapped model to define a length unit")

        variants = variants or {}
        default_variant = variants.get("energy")
        resolved_variants = {
            key: variants.get(key, default_variant)
            for key in [
                "energy",
                "non_conservative_forces",
                "non_conservative_stress",
            ]
        }

        self._energy_key = pick_output("energy", outputs, resolved_variants["energy"])
        self._forces_key = _variant_key(
            "non_conservative_forces", resolved_variants["non_conservative_forces"]
        )
        self._stress_key = _variant_key(
            "non_conservative_stress", resolved_variants["non_conservative_stress"]
        )

        energy_unit = outputs[self._energy_key].unit
        if energy_unit == "":
            raise ValueError("DFTD3 requires the wrapped energy output to define a unit")

        self._force_unit = f"{energy_unit}/{capabilities.length_unit}"
        self._stress_unit = f"{energy_unit}/{capabilities.length_unit}^3"
        self._base_has_forces_output = self._forces_key in outputs
        self._base_has_stress_output = self._stress_key in outputs
        self._cutoff = cutoff
        self._output_dtype = _torch_dtype(capabilities.dtype)

        self._a1 = _as_float(damping_params["a1"], "a1")
        self._a2 = _as_float(damping_params["a2"], "a2")
        self._s8 = _as_float(damping_params["s8"], "s8")
        self._s6 = _as_float(damping_params.get("s6", 1.0), "s6")
        self._k1 = _as_float(damping_params.get("k1", 16.0), "k1")
        self._k3 = _as_float(damping_params.get("k3", -4.0), "k3")
        self._s5_smoothing_on = _as_float(
            damping_params.get("s5_smoothing_on", 1.0e10), "s5_smoothing_on"
        )
        self._s5_smoothing_off = _as_float(
            damping_params.get("s5_smoothing_off", 1.0e10), "s5_smoothing_off"
        )

        rcov = d3_params["rcov"]
        r4r2 = d3_params["r4r2"]
        c6ab = d3_params["c6ab"]
        cn_ref = d3_params["cn_ref"]
        for name, tensor in [
            ("rcov", rcov),
            ("r4r2", r4r2),
            ("c6ab", c6ab),
            ("cn_ref", cn_ref),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"D3 table '{name}' must be a torch.Tensor")

        if rcov.ndim != 1:
            raise ValueError(f"'rcov' must be 1D, got shape {tuple(rcov.shape)}")
        if r4r2.shape != rcov.shape:
            raise ValueError(
                f"'r4r2' must have shape {tuple(rcov.shape)}, got {tuple(r4r2.shape)}"
            )
        if c6ab.ndim != 4:
            raise ValueError(f"'c6ab' must be 4D, got shape {tuple(c6ab.shape)}")
        if cn_ref.shape != c6ab.shape:
            raise ValueError(
                f"'cn_ref' must have shape {tuple(c6ab.shape)}, got {tuple(cn_ref.shape)}"
            )
        if c6ab.shape[0] != rcov.shape[0] or c6ab.shape[1] != rcov.shape[0]:
            raise ValueError(
                "'c6ab' must have shape "
                f"({rcov.shape[0]}, {rcov.shape[0]}, mesh, mesh), got {tuple(c6ab.shape)}"
            )

        max_atomic_type = max(capabilities.atomic_types)
        if max_atomic_type >= rcov.shape[0]:
            raise ValueError(
                "D3 tables do not cover all wrapped-model atomic types: "
                f"maximum atomic type is {max_atomic_type}, but tables only support "
                f"up to {rcov.shape[0] - 1}"
            )

        atomic_types_to_numbers = atomic_types_to_numbers or {}
        type_lookup = torch.zeros(rcov.shape[0], dtype=torch.int32)
        for atomic_type in capabilities.atomic_types:
            number = atomic_types_to_numbers.get(atomic_type, atomic_type)
            if number <= 0:
                raise ValueError(
                    "DFTD3 atomic numbers must be positive, got "
                    f"{number} for atomic type {atomic_type}"
                )
            if number >= rcov.shape[0]:
                raise ValueError(
                    "DFTD3 tables do not cover mapped atomic number "
                    f"{number} for atomic type {atomic_type}"
                )
            type_lookup[atomic_type] = number

        self.register_buffer("_atomic_type_to_number", type_lookup)
        self.register_buffer("_rcov", rcov.detach().to(dtype=torch.float32))
        self.register_buffer("_r4r2", r4r2.detach().to(dtype=torch.float32))
        self.register_buffer("_c6ab", c6ab.detach().to(dtype=torch.float32))
        self.register_buffer("_cn_ref", cn_ref.detach().to(dtype=torch.float32))

        self._d3_neighbor_list = NeighborListOptions(
            cutoff=cutoff,
            full_list=True,
            strict=True,
            requestor="DFTD3",
        )
        self._requested_neighbor_lists = list(model.requested_neighbor_lists())
        if self._d3_neighbor_list not in self._requested_neighbor_lists:
            self._requested_neighbor_lists.append(self._d3_neighbor_list)

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self._requested_neighbor_lists

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return self._requested_inputs

    @staticmethod
    def wrap(
        model: AtomisticModel,
        d3_params: Dict[str, torch.Tensor],
        damping_params: Dict[str, float],
        cutoff: float,
        atomic_types_to_numbers: Optional[Dict[int, int]] = None,
        variants: Optional[Dict[str, Optional[str]]] = None,
        scripting: bool = True,
    ) -> AtomisticModel:
        wrapper = DFTD3(
            model=model.eval(),
            d3_params=d3_params,
            damping_params=damping_params,
            cutoff=cutoff,
            atomic_types_to_numbers=atomic_types_to_numbers,
            variants=variants,
        )

        capabilities = model.capabilities()
        supported_devices = [
            device for device in capabilities.supported_devices if device in ["cpu", "cuda"]
        ]
        if len(supported_devices) == 0:
            raise ValueError(
                "DFTD3 only supports CPU and CUDA devices, but the wrapped model "
                f"declares {capabilities.supported_devices}"
            )

        outputs = capabilities.outputs.copy()
        outputs[wrapper._forces_key] = ModelOutput(
            quantity="force",
            unit=wrapper._force_unit,
            explicit_gradients=[],
            per_atom=True,
        )
        outputs[wrapper._stress_key] = ModelOutput(
            quantity="pressure",
            unit=wrapper._stress_unit,
            explicit_gradients=[],
            per_atom=False,
        )

        new_capabilities = ModelCapabilities(
            outputs=outputs,
            atomic_types=capabilities.atomic_types,
            interaction_range=max(capabilities.interaction_range, cutoff),
            length_unit=capabilities.length_unit,
            supported_devices=supported_devices,
            dtype=capabilities.dtype,
        )

        wrapped_model = AtomisticModel(
            wrapper.eval(),
            model.metadata(),
            capabilities=new_capabilities,
        ).to(device="cpu")

        if scripting:
            wrapped_model = torch.jit.script(wrapped_model)

        return wrapped_model

    def _sorted_neighbor_list(
        self, system: System
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neighbors = system.get_neighbor_list(self._d3_neighbor_list)

        first_atom = neighbors.samples.column("first_atom").to(torch.int64)
        second_atom = neighbors.samples.column("second_atom").to(torch.int32)
        shift_a = neighbors.samples.column("cell_shift_a").to(torch.int32)
        shift_b = neighbors.samples.column("cell_shift_b").to(torch.int32)
        shift_c = neighbors.samples.column("cell_shift_c").to(torch.int32)

        if len(first_atom) == 0:
            empty_ptr = torch.zeros(
                len(system.positions) + 1, dtype=torch.int32, device=system.device
            )
            empty_idx = torch.empty(0, dtype=torch.int32, device=system.device)
            empty_shifts = torch.empty(0, 3, dtype=torch.int32, device=system.device)
            return empty_idx, empty_ptr, empty_shifts

        order = torch.argsort(first_atom)
        first_atom = first_atom[order]
        idx_j = second_atom[order]
        unit_shifts = torch.stack(
            [shift_a[order], shift_b[order], shift_c[order]], dim=1
        )

        counts = torch.bincount(first_atom, minlength=len(system.positions)).to(
            dtype=torch.int32
        )
        neighbor_ptr = torch.zeros(
            len(system.positions) + 1, dtype=torch.int32, device=system.device
        )
        neighbor_ptr[1:] = torch.cumsum(counts, dim=0)

        return idx_j, neighbor_ptr, unit_shifts

    def _compute_d3_correction(
        self,
        system: System,
        compute_stress: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_j, neighbor_ptr, unit_shifts = self._sorted_neighbor_list(system)

        energy = torch.zeros(1, dtype=torch.float32, device=system.device)
        forces = torch.zeros(
            (len(system.positions), 3), dtype=torch.float32, device=system.device
        )
        coord_num = torch.zeros(len(system.positions), dtype=torch.float32, device=system.device)
        if compute_stress:
            virial = torch.zeros((1, 3, 3), dtype=torch.float32, device=system.device)
        else:
            virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=system.device)

        numbers = self._atomic_type_to_number[system.types.to(torch.long)]
        if system.pbc.any():
            torch.ops.nvalchemiops.dftd3_pbc(
                system.positions,
                numbers,
                idx_j,
                neighbor_ptr,
                system.cell.reshape(1, 3, 3),
                unit_shifts,
                self._rcov,
                self._r4r2,
                self._c6ab,
                self._cn_ref,
                self._a1,
                self._a2,
                self._s8,
                energy,
                forces,
                coord_num,
                virial,
                self._k1,
                self._k3,
                self._s6,
                self._s5_smoothing_on,
                self._s5_smoothing_off,
                None,
                compute_stress,
                str(system.device),
            )
        else:
            torch.ops.nvalchemiops.dftd3(
                system.positions,
                numbers,
                idx_j,
                neighbor_ptr,
                self._rcov,
                self._r4r2,
                self._c6ab,
                self._cn_ref,
                self._a1,
                self._a2,
                self._s8,
                energy,
                forces,
                coord_num,
                virial,
                self._k1,
                self._k3,
                self._s6,
                self._s5_smoothing_on,
                self._s5_smoothing_off,
                None,
                str(system.device),
            )

        stress = torch.zeros((3, 3), dtype=system.positions.dtype, device=system.device)
        if compute_stress:
            if not system.pbc.all():
                raise ValueError(
                    "DFTD3 stress output requires periodic boundary conditions in all directions"
                )
            volume = torch.abs(torch.linalg.det(system.cell))
            stress = virial[0].to(dtype=system.positions.dtype) / volume

        return (
            energy[0].to(dtype=system.positions.dtype),
            forces.to(dtype=system.positions.dtype),
            stress,
        )

    def _force_correction_block(
        self, systems: List[System], corrections: List[torch.Tensor], device: torch.device
    ) -> TensorMap:
        samples_list: List[torch.Tensor] = []
        for i, system in enumerate(systems):
            system_column = torch.full(
                (len(system.positions), 1),
                i,
                dtype=torch.int64,
                device=device,
            )
            atom_column = torch.arange(
                len(system.positions), dtype=torch.int64, device=device
            ).reshape(-1, 1)
            samples_list.append(torch.hstack([system_column, atom_column]))

        if len(samples_list) == 0:
            samples_values = torch.empty((0, 2), dtype=torch.int64, device=device)
            values = torch.empty((0, 3, 1), dtype=self._output_dtype, device=device)
        else:
            samples_values = torch.vstack(samples_list)
            values = (
                torch.cat(corrections, dim=0)
                .reshape(-1, 3, 1)
                .to(device=device, dtype=self._output_dtype)
            )

        block = TensorBlock(
            values=values,
            samples=Labels(["system", "atom"], samples_values),
            components=[
                Labels(["xyz"], torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1))
            ],
            properties=Labels(
                ["non_conservative_force"],
                torch.tensor([[0]], dtype=torch.int64, device=device),
            ),
        )
        return TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])

    def _stress_correction_block(
        self, systems: List[System], corrections: List[torch.Tensor], device: torch.device
    ) -> TensorMap:
        if len(systems) == 0:
            values = torch.empty((0, 3, 3, 1), dtype=self._output_dtype, device=device)
            samples_values = torch.empty((0, 1), dtype=torch.int64, device=device)
        else:
            values = (
                torch.stack(corrections, dim=0)
                .reshape(-1, 3, 3, 1)
                .to(device=device, dtype=self._output_dtype)
            )
            samples_values = torch.arange(
                len(systems), dtype=torch.int64, device=device
            ).reshape(-1, 1)

        block = TensorBlock(
            values=values,
            samples=Labels(["system"], samples_values),
            components=[
                Labels(
                    ["xyz_1"],
                    torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
                ),
                Labels(
                    ["xyz_2"],
                    torch.arange(3, dtype=torch.int64, device=device).reshape(-1, 1),
                ),
            ],
            properties=Labels(
                ["non_conservative_stress"],
                torch.tensor([[0]], dtype=torch.int64, device=device),
            ),
        )
        return TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        needs_energy = self._energy_key in outputs
        needs_forces = self._forces_key in outputs
        needs_stress = self._stress_key in outputs

        if selected_atoms is not None and (needs_energy or needs_forces or needs_stress):
            raise NotImplementedError(
                "DFTD3 does not support selected_atoms for corrected outputs"
            )
        if needs_energy and outputs[self._energy_key].per_atom:
            raise NotImplementedError("DFTD3 does not support per-atom corrected energies")

        inner_outputs = torch.jit.annotate(Dict[str, ModelOutput], {})
        for name, output in outputs.items():
            if name == self._forces_key and not self._base_has_forces_output:
                continue
            if name == self._stress_key and not self._base_has_stress_output:
                continue
            inner_outputs[name] = output

        if len(inner_outputs) == 0:
            results = torch.jit.annotate(Dict[str, TensorMap], {})
        else:
            results = self._model(systems, inner_outputs, selected_atoms)

        if not (needs_energy or needs_forces or needs_stress):
            return results

        if len(systems) == 0:
            device = self._rcov.device
        else:
            device = systems[0].device

        energy_corrections: List[torch.Tensor] = []
        force_corrections: List[torch.Tensor] = []
        stress_corrections: List[torch.Tensor] = []

        for system in systems:
            correction_energy, correction_forces, correction_stress = self._compute_d3_correction(
                system,
                compute_stress=needs_stress,
            )
            energy_corrections.append(correction_energy)
            if needs_forces:
                force_corrections.append(correction_forces)
            if needs_stress:
                stress_corrections.append(correction_stress)

        if needs_energy:
            energy_result = results[self._energy_key]
            block = energy_result.block()
            corrected_values = block.values.detach().clone()
            if len(energy_corrections) > 0:
                corrected_values = corrected_values + torch.stack(
                    energy_corrections, dim=0
                ).reshape(-1, 1).to(dtype=corrected_values.dtype, device=device)
            corrected_block = TensorBlock(
                values=corrected_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            results[self._energy_key] = TensorMap(energy_result.keys, [corrected_block])

        if needs_forces:
            correction_map = self._force_correction_block(systems, force_corrections, device)
            if self._base_has_forces_output:
                block = results[self._forces_key].block()
                corrected_block = TensorBlock(
                    values=block.values.detach().clone()
                    + correction_map.block().values.to(
                        dtype=block.values.dtype, device=block.values.device
                    ),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
                results[self._forces_key] = TensorMap(results[self._forces_key].keys, [corrected_block])
            else:
                results[self._forces_key] = correction_map

        if needs_stress:
            correction_map = self._stress_correction_block(systems, stress_corrections, device)
            if self._base_has_stress_output:
                block = results[self._stress_key].block()
                corrected_block = TensorBlock(
                    values=block.values.detach().clone()
                    + correction_map.block().values.to(
                        dtype=block.values.dtype, device=block.values.device
                    ),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
                results[self._stress_key] = TensorMap(results[self._stress_key].keys, [corrected_block])
            else:
                results[self._stress_key] = correction_map

        return results
