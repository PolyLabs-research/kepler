"""
test_package_shims.py - expression, normalize and refine moved into the kepler
package (../src/kepler). The bare-name imports the scripts here use must give
the package modules themselves, so a monkeypatch of either name patches both.
"""
import importlib


def test_bare_names_are_the_package_modules():
    for name in ("expression", "normalize", "refine"):
        assert importlib.import_module(name) is importlib.import_module(f"kepler.{name}")


def test_monkeypatching_the_bare_name_patches_the_package(monkeypatch):
    import kepler.refine
    import refine

    monkeypatch.setattr(refine, "ILL_CONDITIONED_RATIO", 123.0)
    assert kepler.refine.ILL_CONDITIONED_RATIO == 123.0
