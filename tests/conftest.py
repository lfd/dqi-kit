import os
import sys
import warnings

# Make the top-level modules (dqi.py, max_lin_sat.py, ...) importable from tests/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The framework emits many UserWarnings ("Unused variables", "m <= n", ...).
warnings.filterwarnings("ignore")
