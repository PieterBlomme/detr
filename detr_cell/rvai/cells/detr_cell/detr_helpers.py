from pathlib import Path
def get_model_dir():
    cache_dir = Path.home() / ".rvai"
    cache_dir.mkdir(exist_ok=True)

    model_dir = Path(cache_dir) / "models"
    model_dir.mkdir(exist_ok=True)
    model_dir = Path(model_dir) / "detr"
    model_dir.mkdir(exist_ok=True)
    print (model_dir)
    return model_dir
