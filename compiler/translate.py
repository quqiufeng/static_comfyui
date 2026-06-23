#!/usr/bin/env python3
"""StaticPy 翻译器 — Python 静态子集 → Scheme S-表达式
值类型: int=fixnum, float=flonum, bool=boolean
张量: ptr=void* (由 C++ runtime 管理的 GPU/CPU 张量指针)
数组: list[float]=float* (C float 数组, GC 自动回收)
extern fn: 声明 C++ 运行时函数, 编译为 foreign-procedure
"""
import sys
import os
import re
import json
import ast

# ====== C 类型映射 ======
TYPE_MAP = {
    "int": "int",
    "int32": "int",
    "int64": "int",
    "float": "double",
    "double": "double",
    "bool": "boolean",
    "str": "string",
    "ptr": "void*",
    "void": "void",
}

# ====== 外部 FFI 函数注册表 ======
EXTERN_FUNCTIONS = {}

# Class tracking for method dispatch
CLASS_METHODS = {}      # class_name -> set of method names
CLASS_FIELDS = {}       # class_name -> set of field names
CLASS_INSTANCES = {}    # var_name -> class_name (propagated from constructors)
_IN_CLASS = None        # current class name being translated
_IN_METHOD = None       # current method name being translated
DEFINED_FUNCTIONS = set()  # set of static function names (w/o prefix)

# Scheme 预置函数（不是 C 函数，不走 foreign-procedure）
MATH_FUNCTIONS = {
    "sin", "cos", "tan", "log", "log2", "log10", "exp", "sqrt",
    "floor", "ceil", "round", "pow", "fmod",
    "sinh", "cosh", "tanh", "atan2", "asin", "acos", "atan",
    "abs",
}

PRELUDE_FUNCTIONS = {
    "make_float_array", "float_array_set", "float_array_ref", "float_array_offset",
    "float_array_free", "make_ptr_array", "ptr_array_set", "ptr_array_ref",
    "make_int_array", "int_array_set", "int_array_ref",
    "make_dict", "dict_get", "dict_set",
    "file_open", "file_close", "file_read_all", "file_read_floats", "file_write",
    "file_exists", "http_get_simple",
    "parse_json",
    "str_split", "str_join", "str_trim", "str_lower", "str_upper",
    "str_replace", "str_contains", "str_starts_with", "str_ends_with",
    "os_list_dir", "os_getenv", "os_getcwd", "os_file_size",
    "os_file_exists", "os_move_file", "os_delete_file",
    "os_mkdir", "os_rmdir", "os_shell", "os_cwd",
    "string_to_float", "string_to_int",
    "sleep", "clock", "exit_program",
} | MATH_FUNCTIONS


def parse_extern_functions(code):
    """解析 extern fn 声明"""
    pattern = r'extern\s+fn\s+(\w+)\s*\((.*?)\)\s*->\s*(\w+)\s+from\s+"(\w+)"'
    for m in re.finditer(pattern, code, re.DOTALL):
        name = m.group(1)
        if name in PRELUDE_FUNCTIONS:
            continue
        params_str = m.group(2)
        ret = m.group(3)
        lib = m.group(4)
        params = []
        for p in params_str.split(","):
            p = p.strip()
            if p:
                parts = p.split(":")
                pname = parts[0].strip()
                ptype = parts[1].strip() if len(parts) > 1 else "float"
                params.append((pname, ptype))
        EXTERN_FUNCTIONS[name] = {
            "ret": ret,
            "params": params,
            "lib": lib,
        }
    code = re.sub(r'(?m)^\s*extern\s+fn\s+.*?from\s+"\w+"', "", code, flags=re.DOTALL)
    return code


AUGOP_MAP = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/",
    "FloorDiv": "quotient", "Mod": "remainder",
}


def translate_aug_assign(node):
    target = node.target.id if isinstance(node.target, ast.Name) else None
    op = type(node.op).__name__
    val = translate_expr(node.value)
    s_op = AUGOP_MAP.get(op, "+")
    if target:
        return f"(set! {target} ({s_op} {target} {val}))"
    return "(void)"


def has_continue(stmts):
    """Check if a list of statements contains continue (recursively into if bodies)."""
    for s in stmts:
        if isinstance(s, ast.Continue):
            return True
        if isinstance(s, ast.If):
            if has_continue(s.body) or has_continue(s.orelse):
                return True
        if isinstance(s, ast.For):
            if has_continue(s.body):
                return True
        if isinstance(s, ast.While):
            if has_continue(s.body):
                return True
    return False


def translate_type(py_type):
    if py_type.startswith("list[") or py_type == "list":
        return "void*"
    base = py_type.replace("[]", "")
    return TYPE_MAP.get(base, "void*")


def translate_stmt(node, bindings=None):
    if bindings is None:
        bindings = []
    if isinstance(node, ast.FunctionDef):
        return translate_function(node)
    elif isinstance(node, ast.Return):
        val = translate_expr(node.value) if node.value else "#f"
        return val
    elif isinstance(node, ast.AnnAssign):
        target = node.target.id if isinstance(node.target, ast.Name) else None
        if target and node.value:
            val = translate_expr(node.value)
            bindings.append((target, val))
            return f"  {target}"
        return f"  ;; annot-assign {ast.dump(node)}"
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            name = targets[0].id
            val = translate_expr(node.value)
            bindings.append((name, val))
            return f"  {name}"
        return f"  ;; assign {ast.dump(node)}"
    elif isinstance(node, ast.Expr):
        return f"  {translate_expr(node.value)}"
    elif isinstance(node, ast.If):
        test = translate_expr(node.test)
        body = "\n".join(translate_stmt(s) for s in node.body)
        orelse = node.orelse
        if orelse:
            else_body = "\n".join(translate_stmt(s) for s in orelse)
            return f"  (if {test}\n    (begin\n{body})\n    (begin\n{else_body}))"
        else:
            return f"  (if {test}\n    (begin\n{body}))"
    elif isinstance(node, ast.While):
        test = translate_expr(node.test)
        body = "\n".join(translate_stmt(s) for s in node.body)
        return f"  (let loop ()\n    (if {test}\n      (begin\n{body}\n        (loop))))"
    elif isinstance(node, ast.For):
        target = node.target.id if isinstance(node.target, ast.Name) else "i"
        iter_expr = node.iter
        body = "\n".join(translate_stmt(s) for s in node.body)
        if isinstance(iter_expr, ast.Call) and hasattr(iter_expr.func, 'id') and iter_expr.func.id == 'range':
            args = iter_expr.args
            if len(args) == 1:
                n = translate_expr(args[0])
                return f"  (do (({target} 0 (+ {target} 1))) ((>= {target} {n}))\n{body})"
            elif len(args) == 2:
                start = translate_expr(args[0])
                end = translate_expr(args[1])
                return f"  (do (({target} {start} (+ {target} 1))) ((>= {target} {end}))\n{body})"
        return f"  ;; for loop (untranslated)"
    elif isinstance(node, ast.AugAssign):
        return translate_aug_assign(node)
    else:
        return f"  ;; {type(node).__name__}: {ast.dump(node)}"


def translate_block(stmts):
    exprs = []
    for s in stmts:
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            target = None
            val = None
            if isinstance(s, ast.AnnAssign):
                target = s.target.id if isinstance(s.target, ast.Name) else None
                val = translate_expr(s.value) if s.value else "#f"
            else:
                if len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                    target = s.targets[0].id
                    val = translate_expr(s.value)
            if target and val:
                exprs.append(f"(set! {target} {val})")
            else:
                exprs.append(f";; {ast.dump(s)}")
        elif isinstance(s, ast.Expr):
            exprs.append(translate_expr(s.value))
        elif isinstance(s, ast.Return):
            val = translate_expr(s.value) if s.value else "#f"
            exprs.append(val)
        elif isinstance(s, ast.AugAssign):
            exprs.append(translate_aug_assign(s))
        elif isinstance(s, ast.If):
            test = translate_expr(s.test)
            inner_then = translate_block(s.body)
            if s.orelse:
                inner_else = translate_block(s.orelse)
                exprs.append(
                    f"(if {test}\n        (begin {' '.join(inner_then)})\n        (begin {' '.join(inner_else)}))")
            else:
                exprs.append(f"(if {test}\n        (begin {' '.join(inner_then)}))")
        elif isinstance(s, ast.While):
            test = translate_expr(s.test)
            body = translate_block(s.body)
            body_str = wrap_body_with_continue(body, s.body)
            exprs.append(f"(let loop () (if {test} (begin {body_str} (loop))))")
        elif isinstance(s, ast.For):
            target = s.target.id if isinstance(s.target, ast.Name) else "i"
            if isinstance(s.iter, ast.Call) and hasattr(s.iter.func, 'id') and s.iter.func.id == 'range':
                rargs = s.iter.args
                if len(rargs) == 1:
                    n = translate_expr(rargs[0])
                    body_parts = translate_block(s.body)
                    body_str = wrap_body_with_continue(body_parts, s.body)
                    exprs.append(
                        f"(do (({target} 0 (+ {target} 1))) ((>= {target} {n})) {body_str})")
                else:
                    exprs.append(f";; for range with {len(rargs)} args")
            else:
                exprs.append(f";; for loop (untranslated)")
        elif isinstance(s, ast.Call):
            exprs.append(translate_expr(s))
        elif isinstance(s, ast.Pass):
            pass  # no-op
        elif isinstance(s, ast.Global):
            pass  # Python global declaration → Scheme define handles this
        elif isinstance(s, ast.Continue):
            exprs.append("(continue)")  # will be wrapped in call/cc
        else:
            exprs.append(f";; {type(s).__name__}: {ast.dump(s)}")
    return exprs


WARNINGS = []


def warn(msg, node=None):
    if node and hasattr(node, 'lineno'):
        lineno = getattr(node, 'lineno', '?')
        WARNINGS.append(f"  ! {msg} at line {lineno}")
    else:
        WARNINGS.append(f"  ! {msg}")


def translate_expr(node):
    if node is None:
        return "#f"
    if isinstance(node, ast.Constant):
        if node.value is True:
            return "#t"
        if node.value is False:
            return "#f"
        if node.value is None:
            return "#f"
        if isinstance(node.value, str):
            return '"{}"'.format(node.value.replace('"', '\\"'))
        return repr(node.value)
    elif isinstance(node, ast.Name):
        # If this name refers to a user-defined function, add static_ prefix
        # This ensures dict values referencing functions work correctly
        if node.id in DEFINED_FUNCTIONS:
            return f"static_{node.id}"
        return node.id
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return f"(not {translate_expr(node.operand)})"
        if isinstance(node.op, ast.USub):
            return f"(- {translate_expr(node.operand)})"
        return f"({type(node.op).__name__} {translate_expr(node.operand)})"
    elif isinstance(node, ast.BinOp):
        left = translate_expr(node.left)
        right = translate_expr(node.right)
        if isinstance(node.op, ast.Add):
            if (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)) or \
               (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                return f"(string-append {left} {right})"
            return f"(+ {left} {right})"
        elif isinstance(node.op, ast.Sub):
            return f"(- {left} {right})"
        elif isinstance(node.op, ast.Mult):
            return f"(* {left} {right})"
        elif isinstance(node.op, ast.Div):
            return f"(/ {left} {right})"
        elif isinstance(node.op, ast.FloorDiv):
            return f"(quotient {left} {right})"
        elif isinstance(node.op, ast.Mod):
            return f"(mod {left} {right})"
        elif isinstance(node.op, ast.Pow):
            return f"(expt {left} {right})"
        return f"({type(node.op).__name__} {left} {right})"
    elif isinstance(node, ast.Compare):
        left = translate_expr(node.left)
        ops = node.ops
        comparators = node.comparators
        if len(ops) == 1 and len(comparators) == 1:
            right = translate_expr(comparators[0])
            if isinstance(ops[0], ast.Lt):
                return f"(< {left} {right})"
            elif isinstance(ops[0], ast.LtE):
                return f"(<= {left} {right})"
            elif isinstance(ops[0], ast.Gt):
                return f"(> {left} {right})"
            elif isinstance(ops[0], ast.GtE):
                return f"(>= {left} {right})"
            elif isinstance(ops[0], ast.Eq):
                return f"(= {left} {right})"
            elif isinstance(ops[0], ast.NotEq):
                return f"(not (= {left} {right}))"
        return f"(and {left} ... cmp)"
    elif isinstance(node, ast.BoolOp):
        op = "or" if isinstance(node.op, ast.Or) else "and"
        values = [translate_expr(v) for v in node.values]
        return f"({op} {' '.join(values)})"
    elif isinstance(node, ast.Call):
        func = node.func
        args = [translate_expr(a) for a in node.args]
        if isinstance(func, ast.Name):
            name = func.id
            if name in EXTERN_FUNCTIONS:
                return f"({name} {' '.join(args)})"
            if name == "print":
                return f"(print {' '.join(args)})"
            if name == "len":
                return f"(vector-length {args[0]})" if args else "0"
            if name == "range":
                return f"({args[0]})"
            if name == "int":
                if args:
                    arg = args[0]
                    # If the argument to int() is a call to an extern fn that returns int,
                    # the value is already a fixnum — skip truncation
                    if (len(node.args) > 0 and isinstance(node.args[0], ast.Call)
                            and isinstance(node.args[0].func, ast.Name)
                            and node.args[0].func.id in EXTERN_FUNCTIONS
                            and EXTERN_FUNCTIONS[node.args[0].func.id]["ret"] == "int"):
                        return arg
                    # General case: truncate toward zero to match Python semantics
                    return f"(exact (truncate {arg}))"
                return "0"
            if name == "float":
                return f"(inexact {args[0]})" if args else "0.0"
            if name in PRELUDE_FUNCTIONS:
                return f"({name} {' '.join(args)})"
            return f"(static_{name} {' '.join(args)})"
        elif isinstance(func, ast.Attribute):
            obj_expr = func.value
            attr = func.attr
            if isinstance(obj_expr, ast.Name):
                obj_name = obj_expr.id
                # self.method() inside class → dispatch to class method
                if obj_name == 'self' and _IN_CLASS:
                    return f"({_IN_CLASS}_{attr} self {' '.join(args)})"
                # obj.method() for known class instances
                if obj_name in CLASS_INSTANCES:
                    cname = CLASS_INSTANCES[obj_name]
                    return f"({cname}_{attr} {obj_name} {' '.join(args)})"
                # ml.method() — legacy ML module
                if obj_name == "ml":
                    return f"(ml_{attr} {' '.join(args)})"
                # math.method() — map to libm-*
                if obj_name == "math":
                    return f"(libm-{attr} {' '.join(args)})"
            # Complex obj expression: just dispatch as (obj attr args)
            obj_str = translate_expr(obj_expr)
            return f"({obj_str} {attr} {' '.join(args)})"
        return f"({ast.dump(func)} {' '.join(args)})"
    elif isinstance(node, ast.Dict):
        keys = node.keys
        vals = node.values
        if not keys:
            return "(make-dict)"
        items = []
        for k, v in zip(keys, vals):
            key_str = translate_expr(k)
            val_str = translate_expr(v)
            # String keys should be quoted for dict-set!
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                items.append(f'(dict-set! d {key_str} {val_str})')
            else:
                items.append(f'(dict-set! d {key_str} {val_str})')
        items_str = ' '.join(items)
        return f"(let ((d (make-dict))) {items_str} d)"
    elif isinstance(node, ast.List):
        elems = [translate_expr(e) for e in node.elts]
        return f"(vector {' '.join(elems)})"
    elif isinstance(node, ast.Subscript):
        value = translate_expr(node.value)
        slc = node.slice
        idx = translate_expr(slc)
        if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
            return f"(dict-get {value} {idx})"
        return f"(float_array_ref {value} {idx})"
    elif isinstance(node, ast.Attribute):
        obj = node.value
        attr = node.attr
        # self.attr inside class method → Class_attr (prefixed global)
        if isinstance(obj, ast.Name) and obj.id == 'self' and _IN_CLASS:
            return f"{_IN_CLASS}_{attr}"
        # obj.attr where obj is a known class instance → field access
        if isinstance(obj, ast.Name) and obj.id in CLASS_INSTANCES:
            return attr
        # Fallback: dotted pair (for dictionaries or opaque access)
        return f"({translate_expr(node.value)} . {node.attr})"
    elif isinstance(node, ast.JoinedStr):
        parts = []
        for val in node.values:
            if isinstance(val, ast.Constant):
                parts.append(translate_expr(val))
            elif isinstance(val, ast.FormattedValue):
                expr = translate_expr(val.value)
                parts.append(f"(format \"~a\" {expr})")
            else:
                parts.append('""')
        if len(parts) == 1:
            return parts[0]
        return "(string-append " + " ".join(parts) + ")"
    elif isinstance(node, ast.IfExp):
        test = translate_expr(node.test)
        body = translate_expr(node.body)
        orelse = translate_expr(node.orelse)
        return f"(if {test} {body} {orelse})"
    elif isinstance(node, ast.Tuple):
        elts = [translate_expr(e) for e in node.elts]
        return f"(values {' '.join(elts)})"
    else:
        return "(void)"


def wrap_body_with_continue(body_parts, stmts):
    """If the loop body contains continue, wrap in call/cc for skip-to-next-iteration."""
    if has_continue(stmts):
        body_str = " ".join(body_parts)
        return f"(call/cc (lambda (continue) {body_str}))"
    return " ".join(body_parts)


def translate_function(node, source_file=""):
    name = f"static_{node.name}"
    args = [arg.arg for arg in node.args.args]
    # Warn about default parameter values (not supported)
    if node.args.defaults:
        n_defaults = len(node.args.defaults)
        n_args = len(args)
        default_start = n_args - n_defaults
        for i, (arg_name, default_node) in enumerate(zip(args[default_start:], node.args.defaults)):
            default_val = translate_expr(default_node)
            warn(f"Parameter '{arg_name}' has default value {default_val} — caller must provide all args", node)
    # Skip docstring (first string literal expression in body)
    body_start = 0
    if (len(node.body) > body_start and isinstance(node.body[body_start], ast.Expr)
            and isinstance(node.body[body_start].value, ast.Constant)
            and isinstance(node.body[body_start].value.value, str)):
        body_start = 1
    body_exprs = translate_block(node.body[body_start:])
    if len(body_exprs) == 1:
        body = f"  {body_exprs[0]}"
    else:
        exprs_str = "\n    ".join(body_exprs)
        body = f"  (begin\n    {exprs_str}\n  )"
    return f"(define ({name} {' '.join(args)})\n{body})"


def generate_extern_ffi():
    """生成 extern fn 的 Scheme foreign-procedure 声明"""
    lines = []
    loaded_libs = set()
    # .so 路径优先从环境变量读取, 用于 deploy 时指向 lib/ 目录
    torch_std_so = os.environ.get("STATICPY_TORCH_STD_SO",
                                   "cpp_runtime/build/libtorch_std_helper.so")
    for name, info in EXTERN_FUNCTIONS.items():
        lib = info["lib"]
        if lib == "torch_std_helper" and lib not in loaded_libs:
            lines.append(f'(load-shared-object "{torch_std_so}")')
        elif lib == "dgemm_row" and lib not in loaded_libs:
            lines.append('(load-shared-object "/tmp/libdgemm_row.so")')
        loaded_libs.add(lib)
        param_types = []
        for pname, ptype in info["params"]:
            ct = translate_type(ptype)
            param_types.append(ct)
        ret_c = TYPE_MAP.get(info["ret"], "void")
        types_str = " ".join(param_types)
        lines.append(f"(define {name}")
        lines.append(f"  (foreign-procedure \"{name}\" ({types_str}) {ret_c}))")
    return "\n".join(lines)


def translate_class_field_init(node, class_name):
    """Translate self.field = val in __init__ to (set! Class_field val)."""
    target = None
    val = None
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                target = f"{class_name}_{t.attr}"
                val = translate_expr(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Attribute) and isinstance(node.target.value, ast.Name) and node.target.value.id == 'self':
                target = f"{class_name}_{node.target.attr}"
                val = translate_expr(node.value) if node.value else '#f'
    if target and val:
        return f"(set! {target} {val})"
    return None


def translate_stmt_method(node, class_name, field_classes=None):
    """Translate a statement inside a class method body."""
    if field_classes is None:
        field_classes = {}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = None
        val = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                # self.field = val → (set! Class_field val)  (prefix with class name)
                target = f"{class_name}_{t.attr}"
                val = translate_expr(node.value)
            elif isinstance(t, ast.Tuple):
                # Tuple unpacking: a, b = expr
                names = [e.id for e in t.elts if isinstance(e, ast.Name)]
                val = translate_expr(node.value)
                if len(names) == 2 and val:
                    return f"(set! {names[0]} {val})"  # only first for now
                return None
            elif isinstance(t, ast.Name):
                target = t.id
                val = translate_expr(node.value)
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                target = f"{class_name}_{t.attr}"
                val = translate_expr(node.value) if node.value else '#f'
            elif isinstance(t, ast.Name):
                target = t.id
                val = translate_expr(node.value) if node.value else '#f'
        if target and val:
            return f"(set! {target} {val})"
        return None
    elif isinstance(node, ast.Return):
        val = translate_expr(node.value) if node.value else "#f"
        return val
    elif isinstance(node, ast.Expr):
        return translate_expr(node.value)
    elif isinstance(node, ast.AugAssign):
        return translate_aug_assign(node)
    elif isinstance(node, ast.If):
        test = translate_expr(node.test)
        then_parts = [translate_stmt_method(s, class_name) for s in node.body]
        then_str = ' '.join([t for t in then_parts if t])
        if node.orelse:
            else_parts = [translate_stmt_method(s, class_name) for s in node.orelse]
            else_str = ' '.join([e for e in else_parts if e])
            return f"(if {test}\n    (begin {then_str})\n    (begin {else_str}))"
        return f"(if {test}\n    (begin {then_str}))"
    elif isinstance(node, ast.For):
        target = node.target.id if isinstance(node.target, ast.Name) else "i"
        if isinstance(node.iter, ast.Call) and hasattr(node.iter.func, 'id') and node.iter.func.id == 'range':
            rargs = node.iter.args
            body_parts = [translate_stmt_method(s, class_name) for s in node.body]
            body_str = ' '.join([b for b in body_parts if b])
            if len(rargs) == 1:
                n = translate_expr(rargs[0])
                return f"(do (({target} 0 (+ {target} 1))) ((>= {target} {n})) {body_str})"
            elif len(rargs) == 2:
                start = translate_expr(rargs[0])
                end = translate_expr(rargs[1])
                return f"(do (({target} {start} (+ {target} 1))) ((>= {target} {end})) {body_str})"
        return f";; for loop (untranslated)"
    elif isinstance(node, ast.While):
        test = translate_expr(node.test)
        body_parts = [translate_stmt_method(s, class_name) for s in node.body]
        body_str = ' '.join([b for b in body_parts if b])
        return f"(let loop () (if {test} (begin {body_str} (loop))))"
    return f";; {type(node).__name__}: {ast.dump(node)}"


def translate_method_body(node, class_name):
    """Translate class method body — handles self.field patterns."""
    # Track field → class mappings (from constructor calls in __init__)
    field_classes = {}  # field_name -> class_name
    start_idx = 0
    # Skip docstring
    if (len(node.body) > start_idx and isinstance(node.body[start_idx], ast.Expr)
            and isinstance(node.body[start_idx].value, ast.Constant)
            and isinstance(node.body[start_idx].value.value, str)):
        start_idx = 1
    if node.name == '__init__':
        for s in node.body[start_idx:]:

            if isinstance(s, (ast.Assign, ast.AnnAssign)):
                val = None
                target_name = None
                if isinstance(s, ast.Assign) and len(s.targets) == 1:
                    t = s.targets[0]
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                        target_name = t.attr
                        val = s.value
                elif isinstance(s, ast.AnnAssign):
                    if isinstance(s.target, ast.Attribute) and isinstance(s.target.value, ast.Name) and s.target.value.id == 'self':
                        target_name = s.target.attr
                        val = s.value
                if target_name and isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
                    ctor_name = val.func.id
                    if ctor_name in CLASS_METHODS:
                        field_classes[target_name] = ctor_name
    
    body_parts = []
    for s in node.body[start_idx:]:
        if isinstance(s, ast.FunctionDef):
            body_parts.append(translate_function(s))
            continue
        # Pass field_classes to translate_stmt_method
        tr = translate_stmt_method(s, class_name, field_classes)
        if tr:
            body_parts.append(tr)
    if len(body_parts) == 1:
        return f"  {body_parts[0]}"
    body_str = "\n    ".join(body_parts)
    return f"  (begin\n    {body_str}\n  )"


def main():
    code = sys.stdin.read()
    code = parse_extern_functions(code)
    tree = ast.parse(code)
    output_parts = []
    ffi = generate_extern_ffi()
    if ffi:
        output_parts.append(";; extern FFI functions")
        output_parts.append(ffi)
        output_parts.append("")
    source_filename = getattr(tree, 'source_file', '<input>')
    output_parts.append(f";; Source: {source_filename}")
    output_parts.append("")
    output_parts.append(";; StaticPy functions")
    
    # Pass 1: Build CLASS_METHODS and CLASS_INSTANCES maps
    global CLASS_METHODS, CLASS_FIELDS, CLASS_INSTANCES
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cname = node.name
            methods = set()
            fields = set()
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.add(item.name)
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        fields.add(item.target.id)
            CLASS_METHODS[cname] = methods
            CLASS_FIELDS[cname] = fields
    
    # Pass 2: Record all user-defined function names and class instances
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            DEFINED_FUNCTIONS.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name != '__init__':
                    DEFINED_FUNCTIONS.add(f"{node.name}_{item.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = None
            val = None
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value:
                    target = node.target.id
                    val = node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    target = node.targets[0].id
                    val = node.value
            if target and isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
                cname = val.func.id
                if cname in CLASS_METHODS:
                    CLASS_INSTANCES[target] = cname
    
    # Pass 3: Emit translated code
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            output_parts.append(translate_function(node, source_filename))
        elif isinstance(node, ast.ClassDef):
            cname = node.name
            global _IN_CLASS
            _IN_CLASS = cname
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # Class method: rename to ClassName_method, add self param
                    method_name = item.name
                    if method_name == '__init__':
                        # Use full method body translation (handles if/else etc.)
                        body = translate_method_body(item, cname)
                        output_parts.append(f";; {cname} init from __init__")
                        output_parts.append(
                            f"(define ({cname}_init self)\n{body})")
                    else:
                        # Has self as first arg
                        user_args = [a.arg for a in item.args.args if a.arg != 'self']
                        all_params = ' '.join(['self'] + user_args)
                        body = translate_method_body(item, cname)
                        output_parts.append(
                            f"(define ({cname}_{method_name} {all_params})\n{body})")
            _IN_CLASS = None
        elif isinstance(node, ast.Assign):
            # Module-level assignment
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                val = translate_expr(node.value)
                output_parts.append(f"(define {name} {val})")
        elif isinstance(node, ast.AnnAssign):
            # Module-level annotated assignment
            if isinstance(node.target, ast.Name) and node.value:
                name = node.target.id
                val = translate_expr(node.value)
                output_parts.append(f"(define {name} {val})")
    output_parts.append("")
    output_parts.append(";; Entry point")
    output_parts.append("(static_main)")
    print("\n".join(output_parts))
    if WARNINGS:
        print("\n;;=== WARNINGS ===")
        for w in WARNINGS:
            print(f";;{w}")


if __name__ == "__main__":
    main()
