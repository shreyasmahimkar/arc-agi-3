"""Minimal pydantic-v2 shim — ONLY loaded when the real pydantic is absent
(e.g. the Mac->Linux sandbox used for v17 development, which has no network
and so cannot `pip install pydantic`). On the real Mac venv312 the genuine
pydantic is imported instead and this file is never touched.

It implements exactly the subset arcengine.enums uses: BaseModel with
annotated field defaults, Field(default/default_factory/ge/le/...),
PrivateAttr, field_validator (no-op), model_validate, model_dump,
model_dump_json. No validation is enforced — arcengine only relies on the
data-carrier behaviour, never on pydantic's coercion, for offline solving.
"""
from __future__ import annotations
import json as _json


class _FieldInfo:
    __slots__ = ("default", "default_factory", "kw")

    def __init__(self, default=..., default_factory=None, **kw):
        self.default = default
        self.default_factory = default_factory
        self.kw = kw


def Field(default=..., *, default_factory=None, **kw):
    return _FieldInfo(default=default, default_factory=default_factory, **kw)


class _PrivateInfo:
    __slots__ = ("default", "default_factory")

    def __init__(self, default=..., default_factory=None):
        self.default = default
        self.default_factory = default_factory


def PrivateAttr(default=..., *, default_factory=None):
    return _PrivateInfo(default=default, default_factory=default_factory)


def field_validator(*fields, **kw):
    def deco(fn):
        return fn
    return deco


class _ModelMeta(type):
    def __new__(mcls, name, bases, ns):
        cls = super().__new__(mcls, name, bases, ns)
        fields, privates = {}, {}
        for base in bases:
            fields.update(getattr(base, "__fields__", {}))
            privates.update(getattr(base, "__privates__", {}))
        ann = ns.get("__annotations__", {})
        for fname in ann:
            if fname.startswith("_"):
                default = ns.get(fname, ...)
                privates[fname] = default if isinstance(default, _PrivateInfo) else _PrivateInfo(default=default)
            else:
                fields[fname] = ns.get(fname, ...)
        cls.__fields__ = fields
        cls.__privates__ = privates
        return cls


class BaseModel(metaclass=_ModelMeta):
    def __init__(self, **kwargs):
        for fname, default in type(self).__fields__.items():
            if fname in kwargs:
                val = kwargs[fname]
            elif isinstance(default, _FieldInfo):
                if default.default_factory is not None:
                    val = default.default_factory()
                elif default.default is not ...:
                    val = default.default
                else:
                    val = None
            elif default is ...:
                val = None
            elif isinstance(default, (list, dict, set)):
                val = type(default)(default)
            else:
                val = default
            object.__setattr__(self, fname, val)
        for pname, pinfo in type(self).__privates__.items():
            if pinfo.default_factory is not None:
                pval = pinfo.default_factory()
            elif pinfo.default is not ...:
                pval = pinfo.default
            else:
                pval = None
            object.__setattr__(self, pname, pval)

    @classmethod
    def model_validate(cls, data):
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(**data)
        return cls()

    def model_dump(self):
        return {f: getattr(self, f, None) for f in type(self).__fields__}

    def _jsonable(self, v):
        import enum
        if isinstance(v, BaseModel):
            return {k: self._jsonable(getattr(v, k, None)) for k in type(v).__fields__}
        if isinstance(v, enum.Enum):
            return v.value if not isinstance(v.value, tuple) else v.name
        if isinstance(v, (list, tuple)):
            return [self._jsonable(x) for x in v]
        if isinstance(v, dict):
            return {k: self._jsonable(x) for k, x in v.items()}
        try:
            _json.dumps(v)
            return v
        except Exception:
            return str(v)

    def model_dump_json(self, indent=None):
        return _json.dumps(self._jsonable(self), indent=indent)


def install_if_missing():
    """Register this module as `pydantic` in sys.modules iff the real one
    is unavailable. Returns True if the shim was installed."""
    import importlib, sys
    try:
        importlib.import_module("pydantic")
        return False
    except Exception:
        import types
        m = types.ModuleType("pydantic")
        m.BaseModel = BaseModel
        m.Field = Field
        m.PrivateAttr = PrivateAttr
        m.field_validator = field_validator
        sys.modules["pydantic"] = m
        return True
