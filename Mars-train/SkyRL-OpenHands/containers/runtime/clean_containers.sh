#!/bin/bash
log=/var/lib/hourly.log
date +'=== %Y.%m.%d %H:%M ===' >> $log

containers=$(docker container ps --format "{{.ID}} {{.RunningFor}}" | awk '/days/ {print $1} /hours/ {split($2, a, " "); if (a[1] > 2) print $1}' | xargs -r docker rm -f | wc -l)
echo "Cleaned up $containers running containers" >> $log
