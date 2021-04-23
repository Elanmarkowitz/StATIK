"""
This module can be used for profiling functions using the kernprof line_profiler

To use import the profile object from this module.
    from utils.profile import profile

Then decorate the function to profile.

@profile
def func():
    ...

At exit. The profile stats will be printed.
"""



import line_profiler
profile = line_profiler.LineProfiler()
import builtins
builtins.__dict__['profile'] = profile
import atexit
atexit.register(profile.print_stats)