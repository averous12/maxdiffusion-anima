"""Server-private JSON sanitization tests; imports no TPU runtime."""
import ast
import os

SERVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "anima_aesthetic_server.py")


def test_json_public_recursively_drops_private_handles_and_converts_numpy_scalars():
    import numpy as np
    source = open(SERVER, encoding="utf-8").read()
    marker = "def _json_public(value):"
    start = source.index(marker)
    end = source.index("\ndef write_prog", start)
    fn = ast.parse(source[start:end]).body[0]
    module = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np}
    exec(compile(module, SERVER, "exec"), ns)

    class RuntimeArray:
        pass

    value = {"snapref": {"mode": "style", "_kv": (RuntimeArray(), RuntimeArray())},
             "step": np.int32(3), "array": np.float32(0.25)}
    got = ns["_json_public"](value)
    assert got == {"snapref": {"mode": "style"}, "step": 3, "array": 0.25}


if __name__ == "__main__":
    test_json_public_recursively_drops_private_handles_and_converts_numpy_scalars()
    print("SERVER_JSON 1/1")
