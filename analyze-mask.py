from config import settings as conf

dataset = conf.active.dataset
mode = conf.active.mode
relevancy_model = conf.relevancy.active.method
relevancy_threshold = conf.relevancy.active.threshold
mask_dir = (
    "data"
    / dataset
    / "REPP"
    / mode
    / "mask"
    / relevancy_model
    / str(relevancy_threshold)
)
