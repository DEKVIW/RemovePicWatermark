# Freeze-safe torch._numpy._ufuncs patch used by build + runtime.
from __future__ import annotations
from pathlib import Path

MARKER = "# --- RPW_ATTACH_BEGIN ---"

TAIL = "\n# --- RPW_ATTACH_BEGIN ---\n# Freeze-safe attach: module-level for-name loops break under PyInstaller.\n\ndef _rpw_attach_binary(_g, names, impl, deco):\n    for _key in list(names):\n        _g[_key] = deco(getattr(impl, _key))\n\n\ndef _rpw_attach_unary(_g, names, impl, deco):\n    for _key in list(names):\n        _g[_key] = deco(getattr(impl, _key))\n\n\n_rpw_attach_binary(globals(), _binary, _binary_ufuncs_impl, deco_binary_ufunc)\n\n\ndef modf(x, /, *args, **kwds):\n    quot, rem = divmod(x, 1, *args, **kwds)\n    return rem, quot\n\n\n_binary = _binary + [\"divmod\", \"modf\", \"matmul\", \"ldexp\"]\n\n\n# ############# Unary ufuncs ######################\n\n\n_unary = [\n    name\n    for name in dir(_unary_ufuncs_impl)\n    if not name.startswith(\"_\") and name != \"torch\"\n]\n\n\n# these are ufunc(int) -> float\n_fp_unary = [\n    \"arccos\",\n    \"arccosh\",\n    \"arcsin\",\n    \"arcsinh\",\n    \"arctan\",\n    \"arctanh\",\n    \"cbrt\",\n    \"cos\",\n    \"cosh\",\n    \"deg2rad\",\n    \"degrees\",\n    \"exp\",\n    \"exp2\",\n    \"expm1\",\n    \"log\",\n    \"log10\",\n    \"log1p\",\n    \"log2\",\n    \"rad2deg\",\n    \"radians\",\n    \"reciprocal\",\n    \"sin\",\n    \"sinh\",\n    \"sqrt\",\n    \"square\",\n    \"tan\",\n    \"tanh\",\n    \"trunc\",\n]\n\n\ndef deco_unary_ufunc(torch_func):\n    \"\"\"Common infra for unary ufuncs.\n\n    Normalize arguments, sort out type casting, broadcasting and delegate to\n    the pytorch functions for the actual work.\n    \"\"\"\n\n    @normalizer\n    def wrapped(\n        x: ArrayLike,\n        /,\n        out: Optional[OutArray] = None,\n        *,\n        where=True,\n        casting: Optional[CastingModes] = \"same_kind\",\n        order=\"K\",\n        dtype: Optional[DTypeLike] = None,\n        subok: NotImplementedType = False,\n        signature=None,\n        extobj=None,\n    ):\n        if dtype is not None:\n            x = _util.typecast_tensor(x, dtype, casting)\n\n        if torch_func.__name__ in _fp_unary:\n            x = _util.cast_int_to_float(x)\n\n        result = torch_func(x)\n        result = _ufunc_postprocess(result, out, casting)\n        return result\n\n    wrapped.__qualname__ = torch_func.__name__\n    wrapped.__name__ = torch_func.__name__\n\n    return wrapped\n\n\n_rpw_attach_unary(globals(), _unary, _unary_ufuncs_impl, deco_unary_ufunc)\n\n__all__ = _binary + _unary  # noqa: PLE0605\n# --- RPW_ATTACH_END ---\n"


def patch_ufuncs_file(path):
    path = Path(path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if MARKER in text and "_rpw_attach_binary" in text and "def deco_unary_ufunc" in text:
        return False
    end = text.rfind("return quot, rem")
    if end < 0:
        raise ValueError(f"cannot find divmod end in {path}")
    head = text[: end + len("return quot, rem")]
    path.write_text(head + TAIL, encoding="utf-8")
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    return True

if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is None:
        raise SystemExit("usage: patch_torch_ufuncs.py <path-to-_ufuncs.py>")
    changed = patch_ufuncs_file(target)
    print("patched" if changed else "already-ok", target)
