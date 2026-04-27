#!/bin/bash
# Log GPU utilization every 10 seconds for 30 minutes
# Output: shared/logs/gpu_util_<timestamp>.csv

LOGDIR="$(dirname "$0")/../shared/logs"
LOGFILE="$LOGDIR/gpu_util_$(date +%Y%m%d_%H%M%S).csv"
INTERVAL=10
DURATION=1800  # 30 minutes
ITERATIONS=$((DURATION / INTERVAL))

echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,mem_util_pct" > "$LOGFILE"

for i in $(seq 1 $ITERATIONS); do
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
        | while IFS=', ' read -r gpu_util mem_used mem_total; do
            mem_pct=$(awk "BEGIN {printf \"%.1f\", $mem_used/$mem_total*100}")
            echo "$(date +%Y-%m-%dT%H:%M:%S),$gpu_util,$mem_used,$mem_total,$mem_pct"
        done >> "$LOGFILE"
    sleep $INTERVAL
done

echo "Done. Log: $LOGFILE"
# Print summary
echo ""
echo "=== Summary ==="
awk -F',' 'NR>1 {gu+=$2; mu+=$5; n++} END {printf "Samples: %d\nAvg GPU util: %.1f%%\nAvg mem util: %.1f%%\n", n, gu/n, mu/n}' "$LOGFILE"
