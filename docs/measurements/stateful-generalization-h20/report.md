# Stateful expression generalization envelope

Metric: `cuda_ms`. Stable requires variant/control <= 1.25 and paired ratio IQR/median <= 0.1.
All seeds must pass. A failure closes the contiguous tested envelope.
A missing boundary only means the largest tested level passed.

| Family | B/H/K/V/T | Stable tested levels | First unstable/failed |
| --- | --- | --- | --- |
| expression | 1/4/64/64/512 | [0] | 1 |
| expression | 1/4/64/64/2048 | [0] | 1 |
| propagation | 1/4/64/64/512 | [0] | 1 |
| propagation | 1/4/64/64/2048 | [0] | 1 |
| rank | 1/4/64/64/512 | [0] | 1 |
| rank | 1/4/64/64/2048 | [0] | 1 |
| write | 1/4/64/64/512 | [0] | 1 |
| write | 1/4/64/64/2048 | [0] | 1 |
| readout | 1/4/64/64/512 | [0] | 1 |
| readout | 1/4/64/64/2048 | [0] | 1 |
| combined | 1/4/64/64/512 | [0] | 1 |
| combined | 1/4/64/64/2048 | [0] | 1 |
