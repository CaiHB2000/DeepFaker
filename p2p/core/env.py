import os

# 根路径自动解析，可通过环境变量覆盖
ROOT = os.environ.get("DF_ROOT", os.path.abspath(os.path.join(__file__, "../../../")))
DFB_ROOT = os.environ.get("DFB_ROOT", os.path.join(ROOT, "DeepFakeBench"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(DFB_ROOT, "datasets"))
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(ROOT, "results"))

def join(*parts) -> str:
    """安全拼路径"""
    return os.path.join(*parts)
