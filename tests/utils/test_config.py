"""Tests for metroid.utils.config."""

from abc import abstractmethod

import pytest

from metroid.utils.config import Registrable, load_standard_catalogue, load_yaml

# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


def test_load_yaml_parses_mapping(tmp_path):
    """load_yaml parses a YAML document into a dict."""
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb:\n  c: 2\n")
    assert load_yaml(str(path)) == {"a": 1, "b": {"c": 2}}


def test_load_yaml_rejects_non_mapping(tmp_path):
    """A YAML document that is not a mapping raises TypeError."""
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError):
        load_yaml(str(path))


# ---------------------------------------------------------------------------
# load_standard_catalogue (bundled package data)
# ---------------------------------------------------------------------------


def test_load_standard_catalogue_has_rubin():
    """The bundled catalogue includes the rubin standard object."""
    catalogue = load_standard_catalogue()
    assert "rubin" in catalogue
    assert "camera" in catalogue["rubin"]
    assert "pupil" in catalogue["rubin"]
    assert "location" in catalogue["rubin"]


# ---------------------------------------------------------------------------
# Registrable registry dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def hierarchy():
    """A fresh Registrable hierarchy with one registered subclass."""

    class Widget(Registrable, registry_label="widget"):
        @classmethod
        @abstractmethod
        def _from_config(cls, config):
            pass

    class SquareWidget(Widget, type="square"):
        def __init__(self, side):
            self.side = side

        @classmethod
        def _from_config(cls, config):
            return cls(config["side"])

    return Widget, SquareWidget


def test_from_config_dispatches_on_type(hierarchy):
    """from_config builds the subclass named by the 'type' field."""
    Widget, SquareWidget = hierarchy
    widget = Widget.from_config({"type": "square", "side": 3})
    assert isinstance(widget, SquareWidget)
    assert widget.side == 3


def test_from_config_missing_type_raises(hierarchy):
    """A config lacking 'type' raises ValueError with the expected message."""
    Widget, _ = hierarchy
    with pytest.raises(ValueError, match="missing required field 'type'"):
        Widget.from_config({"side": 3})


def test_from_config_unknown_type_raises(hierarchy):
    """An unknown type raises ValueError naming the registry label."""
    Widget, _ = hierarchy
    with pytest.raises(ValueError, match="unknown widget type: triangle"):
        Widget.from_config({"type": "triangle"})


def test_from_config_does_not_mutate_input(hierarchy):
    """from_config copies the config and leaves the caller's dict intact."""
    Widget, _ = hierarchy
    config = {"type": "square", "side": 3}
    Widget.from_config(config)
    assert config == {"type": "square", "side": 3}


def test_sibling_hierarchies_have_separate_registries(hierarchy):
    """A second hierarchy does not share the first's registry."""
    Widget, _ = hierarchy

    class Gadget(Registrable, registry_label="gadget"):
        @classmethod
        @abstractmethod
        def _from_config(cls, config):
            pass

    assert "square" not in Gadget._registry
    assert Gadget._registry_label == "gadget"
