#!/bin/sh
# Staged 80-task benchmark for a free-tier provider.
#
# Free tiers cap requests per day and tokens per minute, so the full matrix
# (4 configs x 3 runs ~= 2,900 requests) does not fit in one day. This script runs
# one config per invocation, in priority order, so progress survives a quota reset:
#
#   sh scripts/run_benchmark_free_tier.sh agent        # headline
#   sh scripts/run_benchmark_free_tier.sh baseline     # the comparison that justifies the agent
#   sh scripts/run_benchmark_free_tier.sh no_verifier  # ablation: numeric hallucination
#   sh scripts/run_benchmark_free_tier.sh no_selfcorrect
#
# Repeat each on later days to build up the 3 runs the README reports as mean +/- std.
# On a paid provider use scripts/run_all_evals.sh instead, which runs everything at once.
set -e
CONFIG="${1:-agent}"
RUNS="${2:-1}"
mkdir -p evaluation/results
LOG="evaluation/results/_benchmark_${CONFIG}.log"
echo "=== $CONFIG x$RUNS starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
python -m evaluation.run_eval --config "$CONFIG" --runs "$RUNS" 2>&1 | tee -a "$LOG"
echo "=== $CONFIG done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
