import sys
from pathlib import Path
import dengine_uciml


def get_config(name: str, default_path: str = ""):
    target = Path(sys.prefix) / f'share/dengine-uciml/configs/{name}'
    if target.exists():
        return target

    root_path = Path(dengine_uciml.__file__).parent.parent
    return root_path / default_path


class BUILTINS:
    class DATASETS:
        HAR_US = get_config('HAR-US.yml', 'configs/datasets/HAR-US.yml')
