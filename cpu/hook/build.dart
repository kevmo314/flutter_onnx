import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final os = input.config.code.targetOS;
    await CBuilder.library(
      name: 'flutter_onnx_cpu',
      assetName: 'flutter_onnx_cpu.dart',
      sources: ['src/matmul.c'],
      frameworks: [if (os == OS.iOS || os == OS.macOS) 'Accelerate'],
    ).run(input: input, output: output);
  });
}
