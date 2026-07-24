# Validation Results

## Software

`software/` contains a CPU rerun of the final deployed checkpoint over all 20,800 EMNIST Letters test samples:

- `summary.json`: total and per-class metrics
- `confusion_matrix.csv`: 26x26 confusion matrix
- `predictions.csv`: one row per test sample

Result: `18,777 / 20,800 = 90.2740%`.

## Board

`board_520/` contains the deterministic class-balanced board run:

- 20 samples per class, 520 total
- selection seed `20260725`
- `summary.txt`: aggregate board metrics
- `predictions.csv`: PyTorch, raw ViT board output, optional postprocessor output and timing
- `uart.log`: raw board UART evidence

The reported accelerator metrics use `board_base_pred`, the raw ViT classifier output before the optional PS pair-confusion postprocessor:

```text
board ViT accuracy             473 / 520 = 90.96%
board vs PyTorch agreement     519 / 520 = 99.81%
average end-to-end latency                 42.55 ms
average PL three-layer time                37.27 ms
```

The optional postprocessor changed one correct sample to an incorrect class, reducing its own output accuracy to 90.77%. It is retained in the source as an experiment but is not counted as an accelerator improvement.
