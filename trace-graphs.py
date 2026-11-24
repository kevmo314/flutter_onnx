from pathlib import Path

path = Path("packages/on_device_tts/lib/src/onnx/graph_worker.dart")
s = path.read_text()
call = "final outputs = await graph.runAsync(feeds, backend: backend);"
assert s.count(call) == 1
path.write_text(s.replace(call, "final timer = Stopwatch()..start();\n      " + call + """
      print('GRAPH_US ${File(path).uri.pathSegments.last} ${timer.elapsedMicroseconds}');"""))
