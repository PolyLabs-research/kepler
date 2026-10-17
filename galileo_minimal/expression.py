"""
expression.py - Compatibility shim: this module now lives in the installable `kepler`
package (src/kepler/expression.py).

The scripts in this folder import it by its bare name. The shim makes
`import expression` return the package module itself (not a copy), so a test that
monkeypatches `expression.<name>` patches `kepler.expression.<name>` too. Without an
installed `kepler` (`pip install -e ..`) the repository's src/ is put on the
path, so running from inside this folder keeps working.
"""
import sys
from pathlib import Path

try:
    import kepler.expression as _module
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    # A run from the repository's parent folder finds the repository itself as a
    # namespace package called kepler; drop it so src/kepler is found instead.
    if getattr(sys.modules.get("kepler"), "__file__", None) is None:
        sys.modules.pop("kepler", None)
    import kepler.expression as _module

sys.modules[__name__] = _module
