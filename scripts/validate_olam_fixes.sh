#!/bin/bash

# Validate OLAM fixes across known-working domains

set -e  # Exit on error

# Clear OLAM workspace
echo "Cleaning OLAM workspace..."
rm -rf /home/omer/projects/OLAM/Analysis/run_*

# Track results
SUCCESSES=0
FAILURES=0
FAILED_DOMAINS=()

# Test each domain
DOMAINS=("blocksworld" "depots" "rover")

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "=== Testing $domain ==="

    OUTPUT_DIR="results/validation_fixes/$domain"

    # Run the analysis
    /usr/bin/python3 scripts/analyze_olam_results.py \
        --domain "benchmarks/olam-compatible/$domain/domain.pddl" \
        --problem "benchmarks/olam-compatible/$domain/p01.pddl" \
        --checkpoints 5 10 \
        --max-iterations 15 \
        --output-dir "$OUTPUT_DIR" > /dev/null 2>&1

    # Check if summary.json was created
    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        # Special check for depots - verify metrics are non-zero
        if [ "$domain" == "depots" ]; then
            # Count how many of the 4 main metrics are exactly 0.0
            ZERO_COUNT=$(grep -E '"final_(safe|complete)_(precision|recall)": 0\.0' "$OUTPUT_DIR/summary.json" | wc -l)
            if [ "$ZERO_COUNT" -lt 4 ]; then
                # At least one metric is non-zero
                echo "✅ $domain: summary.json created, metrics non-zero"
                SUCCESSES=$((SUCCESSES + 1))
            else
                echo "❌ $domain: summary.json created but all metrics are zero"
                FAILURES=$((FAILURES + 1))
                FAILED_DOMAINS+=("$domain")
            fi
        else
            echo "✅ $domain: summary.json created"
            SUCCESSES=$((SUCCESSES + 1))
        fi
    else
        echo "❌ $domain: summary.json NOT created"
        FAILURES=$((FAILURES + 1))
        FAILED_DOMAINS+=("$domain")
    fi
done

# Print summary
echo ""
echo "============================================"
if [ $FAILURES -eq 0 ]; then
    echo "All 3 domains validated successfully!"
else
    echo "Validation complete: $SUCCESSES succeeded, $FAILURES failed"
    echo "Failed domains: ${FAILED_DOMAINS[@]}"
    exit 1
fi
