# Parameter sets

`so101-example.json` is a synthetic API-test center (damping ×1.05, command latency10ms).
It was NOT identified from hardware. Replace it with calibrated/estimated values before
claiming sim-real fidelity. The model parameters are relative to the pinned nominal MJCF.
Changing a center takes effect at the next episode reset; saved active parameters must
not be mistaken for the original center if randomization is enabled.
