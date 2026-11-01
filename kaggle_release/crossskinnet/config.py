"""Read saved JSON configs without YAML's scientific-notation type changes."""
import json
from pathlib import Path
import yaml


def read_config(path):
    text=Path(path).read_text(encoding='utf8')
    try:
        config=json.loads(text)
    except json.JSONDecodeError:
        config=yaml.safe_load(text)
    if not isinstance(config,dict):
        raise ValueError('Configuration must be a JSON/YAML mapping')
    return config
