"""
Run in galileo_minimal.
Print the full LinearODEGenerator.generate() source so we can see
exactly where t and the trajectory are constructed.
"""
import inspect, sys
sys.path.insert(0, ".")
from generators_dynamical import LinearODEGenerator
print(inspect.getsource(LinearODEGenerator))