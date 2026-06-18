"""Dump toto2.forecast() internals to see how it picks its device. Run on SERVER."""
import inspect
import torch

import toto2
from toto2 import Toto2Model

print("toto2 file:", toto2.__file__)

m = Toto2Model.from_pretrained("Datadog/Toto-2.0-313m")
print("model class:", type(m).__module__, type(m).__name__)

# Where the params live before any .to():
print("param device (pre-.to):", next(m.parameters()).device)

fn = m.forecast
print("\nforecast defined in:", inspect.getsourcefile(fn))
try:
    print("----- forecast() source -----")
    print(inspect.getsource(fn))
except OSError as e:
    print("(could not get source:", e, ")")

# Move to GPU and confirm where params end up:
if torch.cuda.is_available():
    m.to("cuda").eval()
    print("\nparam device (post .to cuda):", next(m.parameters()).device)
    # Some HF/custom models stash a device attr the forecast path reads:
    for attr in ("device", "_device", "model_device"):
        if hasattr(m, attr):
            print(f"m.{attr} = {getattr(m, attr)}")
