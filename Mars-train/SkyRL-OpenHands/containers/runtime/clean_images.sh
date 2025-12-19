#!/bin/bash
log=/var/lib/prune.log
date +'=== %Y.%m.%d %H:%M ===' >> $log
# Define your threshold percentage (e.g., 85 means prune when disk is 85% full)
THRESHOLD=85

# Get Docker data root directory
DATA_ROOT=$(docker info --format '{{.DockerRootDir}}')

echo $DATA_ROOT >> $log
# Get filesystem where Docker data is stored
FS=$(df -P "$DATA_ROOT" | tail -n 1 | awk '{print $1}')

# Get current usage percentage
USAGE=$(df -P "$FS" | tail -n 1 | awk '{print $5}' | tr -d '%')

if [ "$USAGE" -gt "$THRESHOLD" ]; then
    echo "Storage usage at ${USAGE}%, exceeding threshold of ${THRESHOLD}%. Pruning..." >> $log
    # Prune everything
    docker system prune -af --filter "until=$((2*24))h"

    USAGE=$(df -P "$FS" | tail -n 1 | awk '{print $5}' | tr -d '%')
    if [ "$USAGE" -gt "$THRESHOLD" ]; then
        # prune even more
        docker system prune -af --filter "until=$((24))h"
    fi
    # Log the results
    echo "Pruning complete. New usage: $(df -P "$FS" | tail -n 1 | awk '{print $5}')" >> $log
else
    echo "Storage usage at ${USAGE}%, below threshold of ${THRESHOLD}%. No pruning necessary." >> $log
fi
