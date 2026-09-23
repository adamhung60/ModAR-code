"""Config path helpers for the public ModAR configurations."""
from pathlib import Path

CONFIG_ROOT = Path("conf")

SEARCH_DIRS = (
    CONFIG_ROOT / "methods",
    CONFIG_ROOT / "examples",
    CONFIG_ROOT,
)


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a leaf name or relative path to an existing config file."""
    p = Path(config_path)
    if p.is_file():
        return p
    if p.parent != Path("."):
        raise FileNotFoundError(f"config not found: {p}")
    name = p.name if p.name.endswith(".yaml") else f"{p.name}.yaml"
    matches = [d / name for d in SEARCH_DIRS if (d / name).is_file()]
    if len(matches) > 1:
        # A bare leaf must not silently select between method and example configs.
        options = ", ".join(str(m) for m in matches)
        raise FileNotFoundError(
            f"config {config_path!r} is ambiguous; pass a full path. "
            f"Candidates: {options}")
    if matches:
        return matches[0]
    searched = ", ".join(str(d / name) for d in SEARCH_DIRS)
    raise FileNotFoundError(
        f"config {config_path!r} not found; searched: {searched}")


def load_config(config_path: str | Path, cli=None):
    """Load a config, merge over its ``base:`` chain, and apply CLI overrides."""
    from omegaconf import OmegaConf

    def _load_chain(path):
        raw = OmegaConf.load(str(path))
        base_path = raw.pop("base", None)
        if base_path is not None:
            raw = OmegaConf.merge(_load_chain(resolve_config_path(base_path)), raw)
        return raw

    merged = _load_chain(resolve_config_path(config_path))
    return OmegaConf.merge(merged, cli) if cli is not None else merged
