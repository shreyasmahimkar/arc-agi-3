"""Spawn-safe shim: expose ../v13/my_agent.py as the importable module
`v13_agent`.

Why this exists: v15's pass-1 BFS reuses v13's BFSSolver, whose macOS
multiprocess expansion pool uses the *spawn* context. Spawned children
unpickle worker objects by re-importing their module BY NAME — a module
materialized via importlib.spec in the parent's memory doesn't exist for
them (observed: every worker died with ModuleNotFoundError: 'v13_agent').
A real file on sys.path fixes it: parent and children alike import this
shim, which executes v13's my_agent.py into this module's namespace.

(We can't simply `import my_agent` from the v13 dir — that name is taken
by v15's own my_agent, which shadows it on sys.path.)
"""
import os as _os

_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__),
                                       '..', 'v13', 'my_agent.py'))
if not _os.path.exists(_path):
    raise ImportError(f"v13 my_agent.py not found at {_path}")
exec(compile(open(_path).read(), _path, 'exec'))
