import os

import ase.build
import metatomic_lj_test
import numpy as np
import pytest
import torch

pytest.importorskip("nvalchemiops")

from metatomic.torch import DFTD3, ModelOutput, load_atomistic_model
from metatomic.torch.ase_calculator import MetatomicCalculator


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729
ATOMIC_NUMBER = 28


def _model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=ATOMIC_NUMBER,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=True,
    )


def _atoms():
    np.random.seed(0xDEADBEEF)
    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)
    return atoms


def _d3_params():
    size = ATOMIC_NUMBER + 1

    rcov = torch.zeros(size, dtype=torch.float32)
    rcov[1:] = 1.0
    rcov[ATOMIC_NUMBER] = 1.5

    r4r2 = torch.zeros(size, dtype=torch.float32)
    r4r2[1:] = 1.0
    r4r2[ATOMIC_NUMBER] = 2.0

    c6ab = torch.zeros((size, size, 5, 5), dtype=torch.float32)
    cn_ref = torch.zeros((size, size, 5, 5), dtype=torch.float32)
    c6ab[ATOMIC_NUMBER, ATOMIC_NUMBER] = 2.0
    cn_ref[ATOMIC_NUMBER, ATOMIC_NUMBER] = 1.0

    return {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6ab": c6ab,
        "cn_ref": cn_ref,
    }


def _damping_params():
    return {
        "a1": 0.4,
        "a2": 4.0,
        "s8": 1.0,
    }


def test_dftd3_wrap_scripts_and_corrects_energy(tmpdir):
    atoms = _atoms()
    base_model = _model()
    scripted_model = DFTD3.wrap(
        base_model,
        d3_params=_d3_params(),
        damping_params=_damping_params(),
        cutoff=CUTOFF,
        scripting=True,
    )

    assert "non_conservative_forces" in scripted_model.capabilities().outputs
    assert "non_conservative_stress" in scripted_model.capabilities().outputs

    requested = {"energy": ModelOutput(per_atom=False)}
    base_calc = MetatomicCalculator(
        base_model,
        check_consistency=True,
        uncertainty_threshold=None,
    )
    scripted_calc = MetatomicCalculator(
        scripted_model,
        check_consistency=True,
        uncertainty_threshold=None,
    )

    base_energy = float(base_calc.run_model(atoms, requested)["energy"].block().values.item())
    scripted_energy = float(
        scripted_calc.run_model(atoms, requested)["energy"].block().values.item()
    )
    assert abs(scripted_energy - base_energy) > 1e-8

    exportable_model = DFTD3.wrap(
        _model(),
        d3_params=_d3_params(),
        damping_params=_damping_params(),
        cutoff=CUTOFF,
        scripting=False,
    )
    path = os.path.join(tmpdir, "wrapped-dftd3.pt")
    exportable_model.save(path)
    reloaded_model = load_atomistic_model(path)
    reloaded_calc = MetatomicCalculator(
        reloaded_model,
        check_consistency=True,
        uncertainty_threshold=None,
    )
    reloaded_energy = float(
        reloaded_calc.run_model(atoms, requested)["energy"].block().values.item()
    )
    assert np.isclose(reloaded_energy, scripted_energy)


def test_dftd3_non_conservative_outputs_drive_compute_energy():
    atoms = _atoms()
    wrapped_model = DFTD3.wrap(
        _model(),
        d3_params=_d3_params(),
        damping_params=_damping_params(),
        cutoff=CUTOFF,
        scripting=True,
    )

    run_calc = MetatomicCalculator(
        wrapped_model,
        check_consistency=True,
        uncertainty_threshold=None,
    )
    outputs = run_calc.run_model(
        atoms,
        {
            "non_conservative_forces": ModelOutput(per_atom=True),
            "non_conservative_stress": ModelOutput(per_atom=False),
        },
    )

    expected_forces = (
        outputs["non_conservative_forces"].block().values.squeeze(-1).detach().cpu().numpy()
    )
    expected_forces = expected_forces - expected_forces.mean(axis=0, keepdims=True)
    expected_stress = (
        outputs["non_conservative_stress"]
        .block()
        .values[0]
        .squeeze(-1)
        .detach()
        .cpu()
        .numpy()
    )

    calc = MetatomicCalculator(
        wrapped_model,
        check_consistency=True,
        non_conservative=True,
        uncertainty_threshold=None,
    )
    results = calc.compute_energy(atoms, compute_forces_and_stresses=True)

    assert np.allclose(results["forces"], expected_forces, atol=1e-6, rtol=1e-6)
    assert np.allclose(results["stress"], expected_stress, atol=1e-6, rtol=1e-6)


def test_dftd3_rejects_per_atom_corrected_energy():
    wrapped_model = DFTD3.wrap(
        _model(),
        d3_params=_d3_params(),
        damping_params=_damping_params(),
        cutoff=CUTOFF,
        scripting=False,
    )

    calc = MetatomicCalculator(
        wrapped_model,
        check_consistency=True,
        uncertainty_threshold=None,
    )
    with pytest.raises(NotImplementedError, match="per-atom corrected energies"):
        calc.run_model(_atoms(), {"energy": ModelOutput(per_atom=True)})
