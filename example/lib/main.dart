import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:flutter_onnx_accelerators/flutter_onnx_accelerators.dart';

void main() => runApp(const MaterialApp(home: XorExample()));

class XorExample extends StatefulWidget {
  const XorExample({super.key});
  @override
  State<XorExample> createState() => _XorExampleState();
}

class _XorExampleState extends State<XorExample> {
  OnnxSession? session;
  MatMulBackend? backend;
  bool left = false, right = true;
  String? status, score;
  int generation = 0;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final asset = await rootBundle.load('assets/xor.onnx');
      session = OnnxSession.fromBytes(
        asset.buffer.asUint8List(asset.offsetInBytes, asset.lengthInBytes),
      );
      final device = await createMatMulBackend();
      if (!mounted) {
        await device?.close();
        return;
      }
      backend = device;
      await _predict();
    } catch (e) {
      if (mounted) setState(() => status = e.toString());
    }
  }

  Future<void> _predict() async {
    final ticket = ++generation;
    var used = false;
    try {
      final outputs = await session!.runAsync(
        {
          'bits': Tensor.float32([1, 2], [left ? 1 : 0, right ? 1 : 0]),
        },
        backend: backend,
        onAccelerated: (_) => used = true,
      );
      if (mounted && ticket == generation) {
        setState(() {
          score = outputs['xor']!.values.single.toStringAsFixed(0);
          status = used ? backend!.name : 'Dart CPU';
        });
      }
    } catch (e) {
      if (mounted && ticket == generation)
        setState(() => status = e.toString());
    }
  }

  @override
  void dispose() {
    backend?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('ONNX inference')),
    body: Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'XOR network',
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              const SizedBox(height: 16),
              SwitchListTile(
                title: const Text('First bit'),
                value: left,
                onChanged: score == null
                    ? null
                    : (v) {
                        setState(() => left = v);
                        _predict();
                      },
              ),
              SwitchListTile(
                title: const Text('Second bit'),
                value: right,
                onChanged: score == null
                    ? null
                    : (v) {
                        setState(() => right = v);
                        _predict();
                      },
              ),
              const Divider(),
              if (score != null)
                Text(
                  'Output  $score',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
              Text(status ?? 'Loading model…'),
            ],
          ),
        ),
      ),
    ),
  );
}
