#!/bin/bash
# Monitor full experiment progress

echo "=== Experiment Progress Monitor ==="
echo "Time: $(date)"
echo ""

echo "Completed Experiments:"
grep -c "Completed in" results/infogain_full_run.log || echo "0"

echo ""
echo "Current Experiment:"
grep -E "^\[.*\]" results/infogain_full_run.log | tail -1

echo ""
echo "Last Iteration:"
grep "iteration [0-9]" results/infogain_full_run.log | tail -1

echo ""
echo "Log File Status:"
ls -lh results/infogain_full_run.log
echo "Last Modified: $(stat -c %y results/infogain_full_run.log)"

echo ""
echo "Process Status:"
ps aux | grep "run_full_experiments" | grep -v grep | awk '{print "CPU: "$3"% | Memory: "$4"% | Time: "$10}'

echo ""
echo "Failed Experiments:"
grep -c "Failed:" results/infogain_full_run.log || echo "0"
