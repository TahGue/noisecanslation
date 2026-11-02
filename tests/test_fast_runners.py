"""The CUDA-graph runners must reproduce their model's own forward pass.

Random-initialised networks keep these tests free of downloads; they need a GPU.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_tiger_runner_matches_tiger_forward():
    from voiceonly.separators.fast_tiger import TigerGraphRunner
    from voiceonly.separators.vendor.tiger_dnr import TIGERDNR

    torch.manual_seed(0)
    model = TIGERDNR(num_blocks=1).cuda().eval()
    x = (np.random.default_rng(0).standard_normal((44100, 1)) * 0.1).astype(np.float32)
    runner = TigerGraphRunner(model, x.shape[0], 1, half=False)
    assert runner.graph is not None
    fast = runner(x)
    with torch.inference_mode():
        t = torch.from_numpy(x.T.copy()).cuda()[:, None]
        ref = torch.stack([model.dialog(t)[:, 2], model.music(t)[:, 0], model.effect(t)[:, 1]])
    assert _rel(fast, ref.cpu().numpy().transpose(0, 2, 1)) < 1e-3  # TF32 noise


def test_mrx_runner_matches_mrx_forward():
    from voiceonly.separators.fast_mrx import MRXGraphRunner
    from voiceonly.separators.vendor.mrx import MRX

    torch.manual_seed(0)
    model = MRX().cuda().eval()
    x = (np.random.default_rng(1).standard_normal((44100, 2)) * 0.1).astype(np.float32)
    runner = MRXGraphRunner(model, x.shape[0], 2, half=False)
    fast = runner(x)
    with torch.inference_mode():
        ref = model(torch.from_numpy(x.T.copy()).cuda()).cpu().numpy().transpose(0, 2, 1)
    assert _rel(fast, ref) < 1e-4
