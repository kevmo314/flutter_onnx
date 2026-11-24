"""Generate small, schema-checked models for Dart VM and browser tests."""

import base64
from pathlib import Path

import onnx
from onnx import TensorProto as T, helper as h, external_data_helper as external

models = {}


def save(name, nodes, inputs, outputs, weights=()):
    graph = h.make_graph(nodes, name, inputs, outputs, initializer=weights)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 28)])
    if name != "external":  # checker requires external files on disk
        onnx.checker.check_model(model, full_check=True)
    models[name] = base64.b64encode(model.SerializeToString()).decode("ascii")


def info(name, shape, dtype=T.FLOAT):
    return h.make_tensor_value_info(name, dtype, shape)


node = h.make_node
save(
    "containers",
    [
        node("SequenceEmpty", [], ["initial"], dtype=T.FLOAT),
        node("SequenceInsert", ["initial", "a"], ["inserted"]),
        node("SequenceLength", ["inserted"], ["length"]),
        node("SequenceConstruct", ["a", "b", "a"], ["seq"]),
        node("SequenceErase", ["seq", "idx"], ["erased"]),
        node("SequenceAt", ["erased", "idx"], ["at"]),
        node("ConcatFromSequence", ["erased"], ["joined"], axis=0),
        node("Optional", ["b"], ["optional"]),
        node("OptionalGetElement", ["optional"], ["present"]),
    ],
    [info("a", [2]), info("b", [2]), info("idx", [], T.INT64)],
    [
        info("length", [], T.INT64),
        info("at", [2]),
        info("joined", [4]),
        info("present", [2]),
    ],
)

save(
    "pool",
    [
        node("MaxRoiPool", ["x", "rois"], ["roi"], pooled_shape=[2, 2]),
        node("GlobalLpPool", ["x"], ["norm"], p=2),
    ],
    [info("x", [1, 1, 4, 4]), info("rois", [1, 5])],
    [info("roi", [1, 1, 2, 2]), info("norm", [1, 1, 1, 1])],
)

count = 12000
nodes = []
outputs = []
for op in ["RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"]:
    inputs = ["x"] if op.endswith("Like") else []
    attrs = {} if inputs else {"shape": [count]}
    nodes.append(node(op, inputs, [op], seed=12.0, **attrs))
    outputs.append(info(op, [count]))
nodes.extend(
    [
        node("Multinomial", ["logits"], ["classes"], sample_size=count, seed=5.0),
        node("Bernoulli", ["x"], ["coin"], seed=7.0),
        node("Dropout", ["x", "ratio", "training"], ["drop", "mask"], seed=9),
    ]
)
save(
    "random",
    nodes,
    [
        info("x", [count]),
        info("logits", [1, 2]),
        info("ratio", []),
        info("training", [], T.BOOL),
    ],
    outputs
    + [
        info("classes", [1, count], T.INT32),
        info("coin", [count]),
        info("drop", [count]),
        info("mask", [count], T.BOOL),
    ],
)

weight = h.make_tensor("weight", T.FLOAT, [2], bytes(8), raw=True)
external.set_external_data(weight, "weights.bin", offset=3, length=8)
weight.ClearField("raw_data")
save(
    "external",
    [node("Add", ["x", "weight"], ["y"])],
    [info("x", [2])],
    [info("y", [2])],
    [weight],
)


def checked(name, model):
    onnx.checker.check_model(model, full_check=True)
    models[name] = base64.b64encode(model.SerializeToString()).decode("ascii")


# A model-local function forwards a caller-supplied float attribute.
body = node("LeakyRelu", ["input"], ["output"])
body.attribute.add(name="alpha", ref_attr_name="slope", type=onnx.AttributeProto.FLOAT)
imports = [h.make_opsetid("", 28), h.make_opsetid("example", 1)]
func = h.make_function(
    "example", "Activate", ["input"], ["output"], [body], imports, attributes=["slope"]
)
graph = h.make_graph(
    [node("Activate", ["x"], ["y"], domain="example", slope=0.125)],
    "local",
    [info("x", [2])],
    [info("y", [2])],
)
checked("function", h.make_model(graph, opset_imports=imports, functions=[func]))

values = h.make_tensor("w", T.FLOAT, [2], [5.0, -3.0])
indices = h.make_tensor("coords", T.INT64, [2, 2], [0, 1, 1, 0])
graph = h.make_graph(
    [
        node(
            "Constant",
            [],
            ["w"],
            sparse_value=h.make_sparse_tensor(values, indices, [2, 2]),
        ),
        node("Add", ["x", "w"], ["y"]),
    ],
    "sparse",
    [info("x", [2, 2])],
    [info("y", [2, 2])],
)
checked("sparse", h.make_model(graph, opset_imports=imports[:1]))

for dtype, name in [(T.DOUBLE, "double"), (T.INT64, "integer")]:
    save(
        name,
        [node("MatMul", ["x", "w"], ["prod"]), node("Add", ["prod", "bias"], ["out"])],
        [info("x", [1, 2], dtype), info("w", [2, 1], dtype), info("bias", [], dtype)],
        [info("out", [1, 1], dtype)],
    )

save(
    "tied_weights",
    [
        node("Transpose", ["weights"], ["projection"]),
        node("MatMul", ["x", "projection"], ["scores"]),
    ],
    [info("x", ["n", 2])],
    [info("scores", ["n", 3])],
    [h.make_tensor("weights", T.FLOAT, [3, 2], [1.0, -1.0, 2.0, 4.0, -3.0, 2.0])],
)
save(
    "integer_gemm",
    [node("Gemm", ["a", "b", "c"], ["y"], alpha=0.5, beta=-0.25)],
    [info("a", [2, 1], T.INT64), info("b", [1, 2], T.INT64), info("c", [2], T.INT64)],
    [info("y", [2, 2], T.INT64)],
)

save(
    "image_decode",
    [node("ImageDecoder", ["encoded"], ["pixels"])],
    [info("encoded", [None], T.UINT8)],
    [info("pixels", [None, None, 3], T.UINT8)],
)

if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "test" / "supplemental_models.dart"
    path.write_text(
        "// Generated by tool/generate_supplemental.py.\nconst supplementalModels = {\n"
        + "".join(f"  '{k}':\n      '{v}',\n" for k, v in models.items())
        + "};\n"
    )
