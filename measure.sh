#!/usr/bin/env bash
set -euo pipefail
cd rtchat
app=com.rtirl.chat
if [[ "$1" == android ]]; then
  adb install -r build/app/outputs/flutter-apk/app-profile.apk
  device=$(adb get-serialno)
  MODEL_DIR="$RUNNER_TEMP/supertonic-3"
  dir="/data/user/0/$app/app_flutter"
  if adb -s "$device" shell run-as "$app" id >/dev/null 2>&1; then
    owner=(run-as "$app")
  else
    # Profile APKs can disable run-as. The Google APIs CI emulator is userdebug.
    adb -s "$device" root
    adb -s "$device" wait-for-device
    test "$(adb -s "$device" shell id -u | tr -d '\r')" = 0
    owner=()
  fi
  adb -s "$device" shell "${owner[@]}" mkdir -p "$dir/supertonic-3"
  tar -C "$MODEL_DIR" -cf - . | adb -s "$device" shell -T "${owner[@]}" tar -xf - -C "$dir/supertonic-3"
  adb -s "$device" shell "${owner[@]}" rm -f "$dir/playback.jsonl"
  if (( ${#owner[@]} == 0 )); then
    uid=$(adb -s "$device" shell stat -c %u "/data/user/0/$app" | tr -d '\r')
    adb -s "$device" shell chown -R "$uid:$uid" "$dir"
    adb -s "$device" shell restorecon -RF "$dir"
  fi
  # Record runnable CPU time during boot, then give each engine the same
  # quiet-start criterion before measuring its first model load and utterance.
  python3 - <<'PY'
import subprocess, time
def sample():
    t = list(map(int, subprocess.check_output(["adb","shell","head","-1","/proc/stat"], text=True).split()[1:9]))
    return sum(t), t[3] + t[4]
last = sample()
quiet = 0
for i in range(60):
    time.sleep(5)
    current = sample()
    idle = (current[1] - last[1]) / max(1, current[0] - last[0])
    last = current
    print(f"Guest CPU idle {idle:.0%}", flush=True)
    quiet = quiet + 1 if idle >= 0.75 else 0
    if quiet >= 3:
        break
PY
  adb -s "$device" shell am start -n "$app/.MainActivity"
  read_report() { adb -s "$device" shell "${owner[@]}" cat "$dir/playback.jsonl" 2>/dev/null | tr -d '\r'; }
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
    python3 - "$RUNNER_TEMP/playback.jsonl" <<'PYTHON'
import json, sys
with open(sys.argv[1]) as f:
    results = [json.loads(line) for line in f]
rows = [r for r in results if "rtf" in r]
playback = [r for r in results if "first_audio_ms" in r]
if len(rows) == 3:
    for r in playback:
        overhead = r["done_ms"] - r["first_audio_ms"] - rows[0]["audio_ms"]
        print(f"Playback round {r['round']}: {overhead:.0f} ms beyond PCM duration after first audio")
if len(rows) != 3 or any(not 0 < r["rtf"] < 1 for r in rows):
    raise SystemExit(f"Missed the real-time synthesis budget: {rows}")
PYTHON
    exit 0
  fi
  if (( attempt % 6 == 0 )); then cat "$RUNNER_TEMP/playback.jsonl"; fi
  sleep 10
done
exit 1
