import json

import pytest

from poremc import reactive_cli, thermo_cli


@pytest.mark.parametrize("build", [thermo_cli.build, reactive_cli.build])
@pytest.mark.parametrize("region", [None, {"lower": [0, 0, 0], "upper": [1, 1, 1]}])
def test_slab_rejects_periodic_z(build, region):
    config = {
        "cell": [24, 24, 48],
        "temperature": 300,
        "species": [{"name": "Na", "valence": 1}, {"name": "Cl", "valence": -1}],
        "log_activities": [-10, -10],
        "channels": [{"name": "salt", "stoichiometry": [1, 1]}],
        "templates": [{"kind": "ion", "name": "Na", "charge": 1},
                      {"kind": "ion", "name": "Cl", "charge": -1}],
        "forcefield": {"cutoff": 8, "kmax": 3, "coulomb": "ewald", "slab": True,
                       "backend": "reference"},
    }
    if region is not None:
        config["region"] = region
    with pytest.raises(ValueError, match="slab electrostatics require a nonperiodic z region"):
        build(config)


@pytest.mark.parametrize("edge,uid,message", [
    (0, 1, "requires template 0"),
    (0, 99, "does not identify an initial molecule"),
    (-1, 0, "must select a configured reaction"),
])
def test_calibration_rejects_invalid_reaction_target(tmp_path, edge, uid, message):
    config = {
        "cell": [30, 30, 30],
        "templates": [{"kind": "acid", "name": "first"},
                      {"kind": "acid", "name": "second"}],
        "forcefield": {"cutoff": 6, "coulomb": "none", "backend": "reference"},
        "initial": [{"template": 0, "state": 1, "anchor_A": [6, 6, 6]},
                    {"template": 1, "state": 1, "anchor_A": [15, 15, 15]}],
        "reactions": [{"template": i, "state_a": 0, "state_b": 1, "stages": 3}
                      for i in range(2)],
        "calibration": {"edge_index": edge, "uid": uid, "samples_per_window": 3,
                        "decorrelation_moves": 1, "burnin_moves": 0},
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config), encoding="utf8")
    out = tmp_path / "calibration"
    out.mkdir()
    previous_config = out / "config.json"
    previous_config.write_text('{"previous_run": true}', encoding="utf8")
    with pytest.raises(ValueError, match=message):
        reactive_cli.main(["calibrate", str(path), "-o", str(out)])
    assert json.loads(previous_config.read_text(encoding="utf8")) == {"previous_run": True}
    assert not (out / "calibration.json").exists()
    assert not (out / "reaction_windows.npz").exists()
