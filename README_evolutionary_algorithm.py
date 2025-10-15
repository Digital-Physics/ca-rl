# Create a target pattern first
uv run evo_ca.py create_pattern --grid-size 12

# Train with evolutionary algorithm
uv run evo_ca.py train \
  --pattern-file custom_pattern_12x12.npy \
  --generations 500 \
  --population-size 100 \
  --sequence-length 10 \
  --mutation-rate 0.1 \
  --live-plot \
  --plot-freq 10

# Demo the best sequence found
uv run evo_ca.py demo \
  --sequence-file best_sequence.npy \
  --pattern-file custom_pattern_12x12.npy