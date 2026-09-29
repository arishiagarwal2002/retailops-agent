#!/bin/sh
# Full benchmark: every configuration, 3 runs each. Needs ANTHROPIC_API_KEY.
# Run the smoke eval first to measure cost per task before committing to this.
set -e
python -m evaluation.retrieval_eval
python -m evaluation.run_eval --config baseline       --runs 3
python -m evaluation.run_eval --config agent          --runs 3
python -m evaluation.run_eval --config no_selfcorrect --runs 3
python -m evaluation.run_eval --config no_verifier    --runs 3
# Model ablation: set ABLATION_MODEL, e.g. claude-opus-5 (Anthropic), gemini-3.8-flash-lite (Gemini),
# llama-3.3-70b-versatile (Groq). Skipped when unset.
if [ -n "$ABLATION_MODEL" ]; then
  python -m evaluation.run_eval --config agent --runs 3 --model "$ABLATION_MODEL"
fi
echo "see evaluation/results/RESULTS.md"
