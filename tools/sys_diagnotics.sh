

#!/usr/bin/env bash
#
# Usage:
#   ./rt_diagnostics.sh <command> [args...]
#   例如：
#   ./rt_diagnostics.sh python3 ./modules/pme/tools/data_viz.py \
#        so ./modules/pme/test/data/debug_data/parking_0520.db \
#        ./modules/pme/test/data/debug_data/parking_0520-1.json
#
# 脚本会：
#  1. 启动目标程序
#  2. 一旦拿到 PID 并确认进程存在，就立即开始监控

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# 1️⃣ 启动目标程序并获取 PID
"$@" &
TARGET_PID=$!
echo "[INFO] Started target: $@"
echo "[INFO] Target PID: $TARGET_PID"

# 2️⃣ 等待内核真正注册该进程（通常几毫秒）
while ! kill -0 "$TARGET_PID" 2>/dev/null; do
    usleep 5000   # 5 ms 间隔检查
done
echo "[INFO] Target process is running, start monitoring..."

# Usage: sudo ./rt_diagnostics.sh <duration_seconds> <app_pid_or_0_for_none>
# Example: sudo ./rt_diagnostics.sh 60 1234
set -euo pipefail

DURATION=${1:-60}
# TARGET_PID=${2:-0}         # PID of the real-time process to monitor (0 = none)
# OUTDIR="/tmp/rt_diag_$(date +%Y%m%d_%H%M%S)"
# 当前文件路径
OUTDIR="$(dirname "$0")/rt_diag_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "RT diagnostics start -> $OUTDIR (duration ${DURATION}s)"; date > "$OUTDIR/start.txt"
uname -a > "$OUTDIR/uname.txt"
lscpu > "$OUTDIR/lscpu.txt"
cat /proc/cpuinfo > "$OUTDIR/cpuinfo.txt"
cat /proc/meminfo > "$OUTDIR/meminfo.txt"
cat /proc/swaps > "$OUTDIR/swaps.txt"
cat /proc/version > "$OUTDIR/version.txt"

# Check basic tools
TOOLS=(cyclictest trace-cmd perf pidstat iostat vmstat tcpdump ethtool)
MISSING=()
for t in "${TOOLS[@]}"; do
  if ! command -v "$t" >/dev/null 2>&1; then
    MISSING+=("$t")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "Warning: missing tools: ${MISSING[*]}" > "$OUTDIR/missing_tools.txt"
  echo "Install recommended: rt-tests (cyclictest), trace-cmd, perf, sysstat (pidstat/iostat), ethtool, tcpdump"
fi

# Snapshot interrupts and affinities
cat /proc/interrupts > "$OUTDIR/interrupts_before.txt"
for irq in /proc/irq/*/smp_affinity_list; do
  [ -f "$irq" ] && echo "$irq: $(cat $irq)" >> "$OUTDIR/irq_affinity_before.txt"
done

# Snapshot current processes and threads (top-like)
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 200 > "$OUTDIR/top_procs.txt"
# Threads for target PID if given
if [ "$TARGET_PID" -ne 0 ]; then
  mkdir -p "$OUTDIR/pid_${TARGET_PID}"
  ps -T -p "$TARGET_PID" -o pid,tid,psr,pcpu,stime,cmd > "$OUTDIR/pid_${TARGET_PID}/threads_before.txt" || true
fi

# Start vmstat, iostat, pidstat collecting in background
vmstat 1 "$DURATION" > "$OUTDIR/vmstat.txt" &
VMSTAT_PID=$!
iostat -x 1 "$DURATION" > "$OUTDIR/iostat.txt" &
IOSTAT_PID=$!
pidstat -t 1 "$DURATION" > "$OUTDIR/pidstat.txt" &
PIDSTAT_PID=$!

# Collect page-faults via perf stat for target pid (if provided)
if [ "$TARGET_PID" -ne 0 ] && command -v perf >/dev/null 2>&1; then
  perf stat -e page-faults,minor-faults,major-faults -p "$TARGET_PID" sleep "$DURATION" 2> "$OUTDIR/perf_pagefaults.txt" &
  PERF_PF_PID=$!
else
  echo "no perf pagefault monitoring" > "$OUTDIR/perf_pagefaults.txt"
fi

# Collect cyclictest if available (measures scheduling jitter)
if command -v cyclictest >/dev/null 2>&1; then
  # run cyclictest: -m lock memory, -n no-fpu, -p priority, -i interval (us), -D duration (ms), -l loops
  CT_OUT="$OUTDIR/cyclictest.txt"
  cyclictest -m -n -p 80 -i 100 -D $((DURATION*1000)) -l 0 > "$CT_OUT" 2>&1 &
  CYCLICTEST_PID=$!
else
  echo "cyclictest not found" > "$OUTDIR/cyclictest.txt"
fi

# Start perf sched record & perf top sampling (if perf available)
if command -v perf >/dev/null 2>&1; then
  perf record -o "$OUTDIR/perf_sched.data" -a -g -e sched:sched_switch -F 99 sleep "$DURATION" >/dev/null 2>&1 || true &
  PERF_SCHED_PID=$!
  perf top -b -o "$OUTDIR/perf_top.txt" -n 1 -p ${TARGET_PID:-0} >/dev/null 2>&1 &
  PERF_TOP_PID=$!
fi

# Start ftrace (via trace-cmd if present, otherwise try debugfs)
TRACE_FILE="$OUTDIR/trace.dat"
if command -v trace-cmd >/dev/null 2>&1; then
  trace-cmd record -o "$TRACE_FILE" -e sched_switch -e irq_handler_entry -e irq_handler_exit -e softirq_raise -e wakeup -F "$DURATION" >/dev/null 2>&1 &
  TRACE_PID=$!
else
  echo "trace-cmd not found, attempting simple ftrace snapshot" > "$OUTDIR/trace_notice.txt"
  # snapshot of current trace buffer (not continuous)
  if [ -d /sys/kernel/debug/tracing ]; then
    echo 0 > /sys/kernel/debug/tracing/tracing_on || true
    echo > /sys/kernel/debug/tracing/trace
    echo 1 > /sys/kernel/debug/tracing/tracing_on || true
    sleep "$DURATION"
    cat /sys/kernel/debug/tracing/trace > "$OUTDIR/ftrace_trace.txt" || true
    echo 0 > /sys/kernel/debug/tracing/tracing_on || true
  fi
fi

# Collect /proc/<pid>/status minflt/majflt periodically if target provided
if [ "$TARGET_PID" -ne 0 ]; then
  MF_LOG="$OUTDIR/pid_${TARGET_PID}/pagefaults_timeseries.txt"
  echo "timestamp minflt majflt" > "$MF_LOG"
  for i in $(seq 1 "$DURATION"); do
    if [ -r "/proc/$TARGET_PID/stat" ]; then
      read -r _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ minflt majflt _ < <(awk '{print $0}' /proc/$TARGET_PID/stat)
      # /proc/<pid>/stat positions: 10 = minflt, 12 = majflt for some kernels — safer to parse /proc/<pid>/status
      # fallback to status parsing
      minflt=$(awk '/minflt/ {print $2}' /proc/"$TARGET_PID"/status 2>/dev/null || echo 0)
      majflt=$(awk '/majflt/ {print $2}' /proc/"$TARGET_PID"/status 2>/dev/null || echo 0)
      echo "$(date +%s) $minflt $majflt" >> "$MF_LOG"
    fi
    sleep 1
  done &
  PF_PID=$!
fi

# Collect thread stack traces for the target PID periodically (optional)
if [ "$TARGET_PID" -ne 0 ] && command -v gdb >/dev/null 2>&1; then
  ( for i in $(seq 1 $((DURATION/5))); do
      ts=$(date +%s)
      mkdir -p "$OUTDIR/pid_${TARGET_PID}/stacks"
      gdb --batch --pid "$TARGET_PID" -ex "thread apply all bt" > "$OUTDIR/pid_${TARGET_PID}/stacks/stack_$ts.txt" 2>/dev/null || true
      sleep 5
    done ) &
  GDB_STACK_PID=$!
fi

# Capture /proc/interrupts during run (every 1s)
( for i in $(seq 1 "$DURATION"); do
    echo "TS: $(date +%s)" >> "$OUTDIR/interrupts_timeseries.txt"
    cat /proc/interrupts >> "$OUTDIR/interrupts_timeseries.txt"
    sleep 1
  done ) &

# Capture top -H thread snapshot every 1s (for target PID)
if [ "$TARGET_PID" -ne 0 ]; then
  ( for i in $(seq 1 "$DURATION"); do
      ps -T -p "$TARGET_PID" -o pid,tid,psr,pcpu,stime,cmd >> "$OUTDIR/pid_${TARGET_PID}/threads_timeseries.txt"
      sleep 1
    done ) &
fi

# Capture network counters (if interface exists)
if command -v ethtool >/dev/null 2>&1; then
  for ifc in $(ls /sys/class/net | grep -v lo); do
    ethtool -S "$ifc" > "$OUTDIR/ethtool_${ifc}_before.txt" 2>/dev/null || true
  done
fi

# Optionally run tcpdump for a few seconds (non-blocking)
if command -v tcpdump >/dev/null 2>&1; then
  tcpdump -i any -s 128 -w "$OUTDIR/tcpdump.pcap" &
  TCPDUMP_PID=$!
fi

# Wait for backgrounded monitoring to finish
echo "Waiting ${DURATION}s for probes to collect..."
sleep "$DURATION"

# kill background probes gently
for pid in ${VMSTAT_PID:-} ${IOSTAT_PID:-} ${PIDSTAT_PID:-} ${PERF_PF_PID:-} ${CYCLICTEST_PID:-} ${PERF_SCHED_PID:-} ${PERF_TOP_PID:-} ${TRACE_PID:-} ${PF_PID:-} ${GDB_STACK_PID:-} ${TCPDUMP_PID:-}; do
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
done

# Final snapshots
cat /proc/interrupts > "$OUTDIR/interrupts_after.txt"
for irq in /proc/irq/*/smp_affinity_list; do
  [ -f "$irq" ] && echo "$irq: $(cat $irq)" >> "$OUTDIR/irq_affinity_after.txt"
done
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 200 > "$OUTDIR/top_procs_after.txt"
if [ "$TARGET_PID" -ne 0 ]; then
  ps -T -p "$TARGET_PID" -o pid,tid,psr,pcpu,stime,cmd > "$OUTDIR/pid_${TARGET_PID}/threads_after.txt" || true
  if [ -r "/proc/$TARGET_PID/status" ]; then
    cp /proc/"$TARGET_PID"/status "$OUTDIR/pid_${TARGET_PID}/status_end.txt"
  fi
fi

# ethtool after
if command -v ethtool >/dev/null 2>&1; then
  for ifc in $(ls /sys/class/net | grep -v lo); do
    ethtool -S "$ifc" > "$OUTDIR/ethtool_${ifc}_after.txt" 2>/dev/null || true
  done
fi

# Save ftrace trace if trace-cmd used
if [ -f "$TRACE_FILE" ]; then
  mv "$TRACE_FILE" "$OUTDIR/"
fi

# Tar up results
tar -czf "${OUTDIR}.tar.gz" -C "$OUTDIR" "$(basename "$OUTDIR")"
echo "Diagnostics collected to ${OUTDIR}.tar.gz"
