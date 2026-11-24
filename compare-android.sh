#!/usr/bin/env bash
set -euo pipefail
lscpu | grep -E 'Model name|CPU\(s\)'
adb shell 'cat /proc/cpuinfo | head -30'
for engine in migrated original; do
  cp "$RUNNER_TEMP/$engine.apk" rtchat/build/app/outputs/flutter-apk/app-profile.apk
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
  adb logcat -c
  adb logcat -v brief 'flutter:I' '*:S' > "$RUNNER_TEMP/$engine-flutter.log" &
  logger=$!
  echo "ENGINE=$engine"
  bash measure.sh android
  kill "$logger" || true
  cp "$RUNNER_TEMP/playback.jsonl" "$RUNNER_TEMP/$engine-playback.jsonl"
  if [[ "$engine" == migrated ]]; then
    grep GRAPH_US "$RUNNER_TEMP/$engine-flutter.log" | tee "$RUNNER_TEMP/graph-timings.log"
  fi
  adb uninstall com.rtirl.chat
done
