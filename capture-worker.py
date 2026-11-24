from pathlib import Path
path=Path("packages/on_device_tts/lib/src/onnx/graph_worker.dart")
s=path.read_text()
key='final outputs = await graph.runAsync(feeds, backend: backend);'
assert s.count(key)==1
s=s.replace(key,"""final report = File('${File(path).parent.parent.parent.path}/operators.log');
      void record(String line) {
        report.writeAsStringSync('$line\\n', mode: FileMode.append);
      }
      final outputs = await runZoned(
        () => graph.runAsync(feeds, backend: backend),
        zoneSpecification: ZoneSpecification(print: (self,parent,zone,line) => record(line)),
      );""")
key="print('GRAPH_US "
assert s.count(key)==1
path.write_text(s.replace(key,"record('GRAPH_US "))
