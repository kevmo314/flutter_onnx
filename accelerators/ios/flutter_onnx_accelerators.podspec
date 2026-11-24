Pod::Spec.new do |s|
  s.name = 'flutter_onnx_accelerators'
  s.version = '0.1.0'
  s.summary = 'Metal matrix multiplication for flutter_onnx.'
  s.description = s.summary
  s.homepage = 'https://github.com/kevmo314/flutter_onnx'
  s.license = { :type => 'See repository' }
  s.author = 'flutter_onnx contributors'
  s.source = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.frameworks = 'Metal', 'MetalPerformanceShaders'
  s.swift_version = '5.0'
end
