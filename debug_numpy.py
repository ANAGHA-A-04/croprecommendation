import sys, os
print('cwd =', os.getcwd())
print('\n'.join(sys.path))
print('--- attempting import numpy ---')
import importlib
try:
    m = importlib.import_module('numpy')
    print('numpy file:', getattr(m, '__file__', None))
except Exception as e:
    print('ERROR:', e)
    import traceback; traceback.print_exc()
