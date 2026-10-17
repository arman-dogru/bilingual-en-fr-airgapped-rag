"""Pinned model revisions (HF commit SHAs) and run metadata, so every result file records what produced it."""
import os, platform, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root; all data/results paths resolve from here
def path(*parts): return os.path.join(ROOT, *parts)
REVISIONS = {
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "1427fd652930e4ba29e8149678df786c240d8825",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
}
def rev(model):
    if model not in REVISIONS: raise SystemExit(f"error: model {model!r} has no pinned revision in src/config.py")
    return REVISIONS[model]
THREADS = int(os.environ.get("THREADS", "4"))  # fixed thread count: CPU matmul reductions depend on it
def env_info():
    import torch, transformers, sentence_transformers, numpy
    return dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
                transformers=transformers.__version__, sentence_transformers=sentence_transformers.__version__,
                numpy=numpy.__version__, threads=torch.get_num_threads(), revisions=REVISIONS)
