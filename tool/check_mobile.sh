#!/usr/bin/env bash
# Run from example/ after building the platform_check.dart target.
set -euo pipefail
platform=$1
marker=flutter_onnx_check.json
if [[ $platform == android ]]; then
  package=com.example.flutter_onnx_example
  adb install -r build/app/outputs/flutter-apk/app-debug.apk
  adb shell pm clear "$package"
  adb shell am start -n "$package/.MainActivity"
  read_marker() {
    local path
    path=$(adb shell run-as "$package" find . -name "$marker" | tr -d '\r')
    [[ -n "$path" ]] && adb shell run-as "$package" cat "$path"
  }
elif [[ $platform == ios ]]; then
  simulator=$2
  app=build/ios/iphonesimulator/Runner.app
  bundle=$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$app/Info.plist")
  xcrun simctl install "$simulator" "$app"
  data=$(xcrun simctl get_app_container "$simulator" "$bundle" data)
  find "$data" -name "$marker" -delete
  xcrun simctl launch "$simulator" "$bundle"
  read_marker() {
    local path
    path=$(find "$data" -name "$marker" -print -quit)
    [[ -n "$path" ]] && cat "$path"
  }
else
  exit 2
fi
for ((attempt=0; attempt<90; attempt++)); do
  if result=$(read_marker); then
    echo "$result"
    echo "$result" | jq -e '.ok == true and .xorCases == 4 and .attentionOutputs > 0'
    exit 0
  fi
  sleep 2
done
echo "Device did not report inference results" >&2
if [[ $platform == android ]]; then adb logcat -d -s flutter AndroidRuntime; fi
exit 1
