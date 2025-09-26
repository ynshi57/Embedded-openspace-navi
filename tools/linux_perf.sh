#!/bin/bash
#
# Linux Performance Analysis in 60s
# Inspired by Brendan Gregg
#

echo "========================================"
echo "  Linux Performance Analysis in 60s"
echo "  $(date)"
echo "========================================"

echo ""
echo "[1] uptime (load average)"
uptime

echo ""
echo "[2] dmesg (last 5 lines)"
dmesg | tail -n 5

echo ""
echo "[3] vmstat (1s, 5 times)"
vmstat 1 5

echo ""
echo "[4] mpstat -P ALL (1s, 5 times)"
which mpstat >/dev/null 2>&1 && mpstat -P ALL 1 5 || echo "mpstat not installed (yum install sysstat)"

echo ""
echo "[5] pidstat (1s, 5 times)"
which pidstat >/dev/null 2>&1 && pidstat 1 5 || echo "pidstat not installed (yum install sysstat)"

echo ""
echo "[6] iostat -xz (1s, 5 times)"
which iostat >/dev/null 2>&1 && iostat -xz 1 5 || echo "iostat not installed (yum install sysstat)"

echo ""
echo "[7] free -m"
free -m

echo ""
echo "[8] sar -n DEV (1s, 5 times)"
which sar >/dev/null 2>&1 && sar -n DEV 1 5 || echo "sar not installed (yum install sysstat)"

echo ""
echo "[9] sar -n TCP,ETCP (1s, 5 times)"
which sar >/dev/null 2>&1 && sar -n TCP,ETCP 1 5 || echo "sar not installed (yum install sysstat)"

echo ""
echo "[10] top (batch mode, 5 iterations)"
top -b -n 5 | head -20

echo "========================================"
echo "  Done. Check output above for bottlenecks."
echo "========================================"
