package dev.flutteronnx.accelerators;

import android.os.Build;
import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.StandardMethodCodec;

public final class FlutterOnnxAcceleratorsPlugin implements FlutterPlugin, MethodChannel.MethodCallHandler {
    private MethodChannel channel;
    private boolean ready;
    private static native boolean available();
    private static native float[] product(float[] left, float[] right, int rows, int inner, int columns);

    @Override public void onAttachedToEngine(FlutterPluginBinding binding) {
        if (Build.VERSION.SDK_INT >= 29) {
            try {
                System.loadLibrary("flutter_onnx_nnapi");
                ready = available();
            } catch (UnsatisfiedLinkError error) {
                ready = false;
            }
        }
        channel = new MethodChannel(binding.getBinaryMessenger(), "dev.flutteronnx/matmul",
            StandardMethodCodec.INSTANCE, binding.getBinaryMessenger().makeBackgroundTaskQueue());
        channel.setMethodCallHandler(this);
    }
    @Override public void onMethodCall(MethodCall call, MethodChannel.Result result) {
        if (call.method.equals("available")) {
            result.success(ready ? "Android NNAPI" : null);
        } else if (call.method.equals("multiply") && ready) {
            float[] a = call.argument("a"), b = call.argument("b");
            Integer m = call.argument("m"), k = call.argument("k"), n = call.argument("n");
            if (a == null || b == null || m == null || k == null || n == null ||
                m <= 0 || k <= 0 || n <= 0 || (long)m*k != a.length ||
                (long)k*n != b.length || (long)m*n > Integer.MAX_VALUE / 4) {
                result.success(null);
            } else {
                result.success(product(a,b,m,k,n));
            }
        } else if (call.method.equals("multiply")) {
            result.success(null);
        } else {
            result.notImplemented();
        }
    }
    @Override public void onDetachedFromEngine(FlutterPluginBinding binding) {
        channel.setMethodCallHandler(null);
    }
}
