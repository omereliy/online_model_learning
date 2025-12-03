#!/bin/bash

echo "========================================="
echo "EXPERIMENT STATUS CHECK"
echo "========================================="

echo -e "\n📊 COMPLETED DOMAINS:"
ls /home/omer/projects/online_model_learning/results/paper/comparison_20251118_155028/information_gain/ 2>/dev/null | tr '\n' ' '
echo -e "\n"

echo -e "\n📈 SAMPLE RESULTS (Blocksworld p00):"
if [ -f "/home/omer/projects/online_model_learning/results/paper/comparison_20251118_155028/information_gain/blocksworld/p00/metrics.json" ]; then
    echo "Metrics available:"
    python3 -c "import json; data=json.load(open('/home/omer/projects/online_model_learning/results/paper/comparison_20251118_155028/information_gain/blocksworld/p00/metrics.json')); print(f'  Final Precision: {data.get(\"final_precision\", \"N/A\")}'); print(f'  Final Recall: {data.get(\"final_recall\", \"N/A\")}'); print(f'  Iterations: {data.get(\"total_iterations\", \"N/A\")}')" 2>/dev/null || echo "  Could not parse metrics"
fi

echo -e "\n📝 LOG FILE STATUS:"
LOG_FILE="/home/omer/projects/online_model_learning/results/infogain_validated_run.log"
if [ -f "$LOG_FILE" ]; then
    echo "Location: $LOG_FILE"
    echo "Size: $(ls -lh $LOG_FILE | awk '{print $5}')"
    echo "Last modified: $(ls -lh $LOG_FILE | awk '{print $6, $7, $8}')"
    echo "Last 3 lines:"
    tail -3 $LOG_FILE | sed 's/^/  /'
fi

echo -e "\n🏃 PROCESS STATUS:"
if ps aux | grep -q "1777800.*run_full_experiments"; then
    echo "✅ Experiment is still running (PID: 1777800)"
    ps aux | grep "1777800" | grep -v grep | awk '{print "  CPU: " $3 "%, Memory: " $4 "%"}'
else
    echo "❌ Experiment process not found"
fi

echo -e "\n========================================="
echo "To monitor live: tail -f $LOG_FILE"
echo "To browse results: cd /home/omer/projects/online_model_learning/results/paper/comparison_20251118_155028/"