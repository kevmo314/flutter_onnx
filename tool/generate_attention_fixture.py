"""Generate the cached attention fixture for web and mobile platform tests."""

import json
from pathlib import Path
import numpy as np
from onnx import helper as h, TensorProto as T, checker
from onnx.reference import ReferenceEvaluator

root = Path(__file__).resolve().parents[1] / "example/assets"
rng = np.random.default_rng(314)
shapes = {
    "q": [1, 2, 2, 4],
    "k": [1, 1, 2, 4],
    "v": [1, 1, 2, 3],
    "past_k": [1, 1, 1, 4],
    "past_v": [1, 1, 1, 3],
}
outputs = {"y": [1, 2, 2, 3], "present_k": [1, 1, 3, 4], "present_v": [1, 1, 3, 3]}
info = lambda d: [h.make_tensor_value_info(k, T.FLOAT, s) for k, s in d.items()]
node = h.make_node(
    "Attention", ["q", "k", "v", "", "past_k", "past_v"], list(outputs), is_causal=1
)
m = h.make_model(
    h.make_graph([node], "cached_gqa", info(shapes), info(outputs)),
    ir_version=14,
    opset_imports=[h.make_opsetid("", 28)],
)
checker.check_model(m, full_check=True)
feeds = {k: rng.normal(size=s).astype(np.float32) for k, s in shapes.items()}
values = ReferenceEvaluator(m).run(None, feeds)
packed = lambda items: {
    k: {"shape": list(v.shape), "values": v.ravel().tolist()} for k, v in items
}
(root / "attention.onnx").write_bytes(m.SerializeToString())
(root / "attention.json").write_text(
    json.dumps(
        {"feeds": packed(feeds.items()), "outputs": packed(zip(outputs, values))},
        separators=(",", ":"),
    )
    + "\n"
)
