#!/usr/bin/env bash
set -euo pipefail
cd rtchat
app=com.rtirl.chat
if [[ "$1" == android ]]; then
  adb install -r build/app/outputs/flutter-apk/app-profile.apk
  dir=/sdcard/Android/data/$app/files
  adb shell mkdir -p "$dir"
  adb push "$RUNNER_TEMP/supertonic-3" "$dir/"
  adb logcat -c
  adb shell am start -n "$app/.MainActivity"
  read_report() { adb shell cat "$dir/playback.jsonl" 2>/dev/null | tr -d '\r'; }
  logs() { adb logcat -d -t 1200 > "$RUNNER_TEMP/device.log"; }
else
  device=$(xcrun simctl list devices available -j | python3 -c 'import sys,json; print(next(d["udid"] for rt,ds in json.load(sys.stdin)["devices"].items() if ".iOS-" in rt for d in ds if d["name"].startswith("iPhone")))')
  xcrun simctl boot "$device"
  xcrun simctl bootstatus "$device" -b
  flutter build ios --simulator --debug -t integration_test/profile_playback.dart
  xcrun simctl install "$device" build/ios/iphonesimulator/Runner.app
  dir=$(xcrun simctl get_app_container "$device" "$app" data)/Documents
  mkdir -p "$dir"
  cp -R "$RUNNER_TEMP/supertonic-3" "$dir/"
  xcrun simctl launch --stdout="$RUNNER_TEMP/device.log" --stderr="$RUNNER_TEMP/device-error.log" "$device" "$app"
  read_report() { cat "$dir/playback.jsonl" 2>/dev/null; }
  logs() { cat "$RUNNER_TEMP/device-error.log" >> "$RUNNER_TEMP/device.log" || true; }
fi
trap logs EXIT
for (( attempt=0; attempt<180; ++attempt )); do
  read_report > "$RUNNER_TEMP/playback.jsonl" || true
  if grep -q '"error":' "$RUNNER_TEMP/playback.jsonl"; then cat "$RUNNER_TEMP/playback.jsonl"; exit 1; fi
  if grep -q '"complete":true' "$RUNNER_TEMP/playback.jsonl"; then
    cat "$RUNNER_TEMP/playback.jsonl" | tee -a "$GITHUB_STEP_SUMMARY"
    exit 0
  fi
  if (( attempt % 6 == 0 )); then cat "$RUNNER_TEMP/playback.jsonl"; fi
  sleep 10
done
exit 1
