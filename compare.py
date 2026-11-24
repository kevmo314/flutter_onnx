"""Generate two diagnostic builds from the pinned runtime on Apple silicon."""
from pathlib import Path
import sys

mode = sys.argv[1]
p = Path("cpu/lib/flutter_onnx_cpu.dart")
s = p.read_text().replace("Platform.isAndroid || Platform.isLinux", "Platform.isAndroid || Platform.isLinux || Platform.isMacOS || Platform.isIOS")
p.write_text(s)
if mode == "packed":
    p = Path("cpu/src/matmul.c")
    s = p.read_text().replace("#ifndef __APPLE__", "#if 1").replace("#ifdef __APPLE__", "#if 0").replace("#if defined(__linux__)", "#if defined(__linux__) || defined(__APPLE__)")
    p.write_text("#pragma clang fp contract(off)\n" + s)
