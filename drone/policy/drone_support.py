"""Use the simulator checkout bundled with this drone workflow."""
import sys
from paths import SIMULATOR
if str(SIMULATOR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR))
