import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final os = input.config.code.targetOS;
    final apple = os == OS.iOS || os == OS.macOS;
    await CBuilder.library(
      name: 'flutter_onnx_cpu',
      assetName: 'flutter_onnx_cpu.dart',
      sources: [
        'src/matmul.c',
        'src/pointwise.c',
        'src/channelwise.c',
        if (apple) 'src/metal_projection.m',
      ],
      flags: [if (apple) '-fobjc-arc'],
      libraries: [
        if (os == OS.linux) 'pthread',
        if (os == OS.android || os == OS.linux) 'm',
      ],
      frameworks: [
        if (apple) ...[
          'Accelerate',
          'Foundation',
          'Metal',
        ],
      ],
    ).run(input: input, output: output);
  });
}
