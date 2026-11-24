#include <android/NeuralNetworks.h>
#include <cstdint>
#include <dlfcn.h>
#include <jni.h>
#include <memory>
#include <vector>

// Resolve symbols at runtime so the plugin remains loadable on older Android
// releases. Only API 29+ devices with a GPU or accelerator are candidates.
#define NN_FUNCTIONS(F)                                                        \
  F(_getDeviceCount)                                                           \
  F(_getDevice) F(Device_getType) F(Model_create) F(Model_free)                \
      F(Model_addOperand) F(Model_setOperandValue) F(Model_addOperation)       \
          F(Model_identifyInputsAndOutputs) F(Model_finish)                    \
              F(Model_getSupportedOperationsForDevices)                        \
                  F(Compilation_createForDevices) F(Compilation_finish)        \
                      F(Compilation_free) F(Execution_create)                  \
                          F(Execution_free) F(Execution_setInput)              \
                              F(Execution_setOutput) F(Execution_compute)

namespace {
struct Api {
  void *lib = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
#define DECLARE(name) decltype(&ANeuralNetworks##name) name = nullptr;
  NN_FUNCTIONS(DECLARE)
#undef DECLARE
  std::vector<const ANeuralNetworksDevice *> devices;
  Api() {
    if (!lib)
      return;
#define LOAD(name)                                                             \
  name =                                                                       \
      reinterpret_cast<decltype(name)>(dlsym(lib, "ANeuralNetworks" #name));   \
  if (!name)                                                                   \
    return;
    NN_FUNCTIONS(LOAD)
#undef LOAD
    uint32_t count = 0;
    if (_getDeviceCount(&count))
      return;
    for (uint32_t i = 0; i < count; ++i) {
      ANeuralNetworksDevice *device = nullptr;
      int32_t type;
      if (!_getDevice(i, &device) && !Device_getType(device, &type) &&
          (type == ANEURALNETWORKS_DEVICE_GPU ||
           type == ANEURALNETWORKS_DEVICE_ACCELERATOR))
        devices.push_back(device);
    }
  }
};
const Api &api() {
  static const Api instance;
  return instance;
}

bool multiply(const float *a, const float *b, float *out, uint32_t m,
              uint32_t k, uint32_t n) {
  const auto &nn = api();
  if (nn.devices.empty())
    return false;
  ANeuralNetworksModel *model = nullptr;
  if (nn.Model_create(&model))
    return false;
  std::unique_ptr<ANeuralNetworksModel, decltype(nn.Model_free)> ownedModel(
      model, nn.Model_free);
  const uint32_t ad[] = {m, k}, wd[] = {n, k}, bd[] = {n}, od[] = {m, n};
  const ANeuralNetworksOperandType operands[] = {
      {ANEURALNETWORKS_TENSOR_FLOAT32, 2, ad, 0, 0},
      {ANEURALNETWORKS_TENSOR_FLOAT32, 2, wd, 0, 0},
      {ANEURALNETWORKS_TENSOR_FLOAT32, 1, bd, 0, 0},
      {ANEURALNETWORKS_INT32, 0, nullptr, 0, 0},
      {ANEURALNETWORKS_TENSOR_FLOAT32, 2, od, 0, 0}};
  for (const auto &operand : operands)
    if (nn.Model_addOperand(model, &operand))
      return false;
  // FULLY_CONNECTED takes [output channels, input channels] weights.
  std::vector<float> weights(size_t(k) * n), bias(n, 0.f);
  for (uint32_t i = 0; i < k; ++i)
    for (uint32_t j = 0; j < n; ++j)
      weights[size_t(j) * k + i] = b[size_t(i) * n + j];
  const int32_t activation = ANEURALNETWORKS_FUSED_NONE;
  const uint32_t inputs[] = {0, 1, 2, 3}, outputs[] = {4};
  if (nn.Model_setOperandValue(model, 1, weights.data(),
                               weights.size() * sizeof(float)) ||
      nn.Model_setOperandValue(model, 2, bias.data(),
                               bias.size() * sizeof(float)) ||
      nn.Model_setOperandValue(model, 3, &activation, sizeof(activation)) ||
      nn.Model_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, inputs,
                            1, outputs) ||
      nn.Model_identifyInputsAndOutputs(model, 1, inputs, 1, outputs) ||
      nn.Model_finish(model))
    return false;

  for (const auto *device : nn.devices) {
    bool supported = false;
    if (nn.Model_getSupportedOperationsForDevices(model, &device, 1,
                                                  &supported) ||
        !supported)
      continue;
    ANeuralNetworksCompilation *compilation = nullptr;
    if (nn.Compilation_createForDevices(model, &device, 1, &compilation))
      continue;
    std::unique_ptr<ANeuralNetworksCompilation, decltype(nn.Compilation_free)>
        compiled(compilation, nn.Compilation_free);
    if (nn.Compilation_finish(compilation))
      continue;
    ANeuralNetworksExecution *execution = nullptr;
    if (nn.Execution_create(compilation, &execution))
      continue;
    std::unique_ptr<ANeuralNetworksExecution, decltype(nn.Execution_free)> run(
        execution, nn.Execution_free);
    if (!nn.Execution_setInput(execution, 0, nullptr, a,
                               size_t(m) * k * sizeof(float)) &&
        !nn.Execution_setOutput(execution, 0, nullptr, out,
                                size_t(m) * n * sizeof(float)) &&
        !nn.Execution_compute(execution))
      return true;
  }
  return false;
}
} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_flutteronnx_accelerators_FlutterOnnxAcceleratorsPlugin_available(
    JNIEnv *, jclass) {
  return !api().devices.empty();
}
extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_flutteronnx_accelerators_FlutterOnnxAcceleratorsPlugin_product(
    JNIEnv *env, jclass, jfloatArray left, jfloatArray right, jint m, jint k,
    jint n) {
  try {
    std::vector<float> a(env->GetArrayLength(left)),
        b(env->GetArrayLength(right)), c(size_t(m) * n);
    env->GetFloatArrayRegion(left, 0, a.size(), a.data());
    env->GetFloatArrayRegion(right, 0, b.size(), b.data());
    if (env->ExceptionCheck() ||
        !multiply(a.data(), b.data(), c.data(), m, k, n))
      return nullptr;
    auto result = env->NewFloatArray(c.size());
    if (result)
      env->SetFloatArrayRegion(result, 0, c.size(), c.data());
    return result;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
