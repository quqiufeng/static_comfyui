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
            exprs.append(f"(let loop () (if {test} (begin {' '.join(body)} (loop))))")
        elif isinstance(s, ast.Call):
            exprs.append(translate_expr(s))
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
                return f"(exact (round {args[0]}))" if args else "0"
            if name == "float":
                return f"(inexact {args[0]})" if args else "0.0"
            if name in PRELUDE_FUNCTIONS:
                return f"({name} {' '.join(args)})"
            return f"(static_{name} {' '.join(args)})"
        elif isinstance(func, ast.Attribute):
            obj = None
            if isinstance(func.value, ast.Name):
                obj = func.value.id
            if obj:
                attr = func.attr
                if obj == "ml":
                    return f"(ml_{attr} {' '.join(args)})"
            return f"({ast.dump(func)} . {attr})"
        elif isinstance(func, ast.Attribute):
            obj = translate_expr(func.value)
            attr = func.attr
            return f"({obj} {attr} {' '.join(args)})"
        return f"({ast.dump(func)} {' '.join(args)})"
    elif isinstance(node, ast.Dict):
        keys = [translate_expr(k) for k in node.keys]
        vals = [translate_expr(v) for v in node.values]
        if not keys:
            return "(make-dict)"
        items = []
        for k, v in zip(keys, vals):
            items.append(f"(dict-set! d {k} {v})")
        return f"(let ((d (make-dict))) {' '.join(items)} d)"
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


def translate_function(node, source_file=""):
    name = f"static_{node.name}"
    args = [arg.arg for arg in node.args.args]
    all_bindings = []
    body_exprs = []
    si = 0
    lineno = getattr(node, 'lineno', 0)
    comment = f";; {source_file}:{lineno} def {node.name}()"
    while si < len(node.body):
        stmt = node.body[si]
        if isinstance(stmt, ast.FunctionDef):
            body_exprs.append(translate_function(stmt))
        elif isinstance(stmt, ast.Return):
            val = translate_expr(stmt.value) if stmt.value else "#f"
            body_exprs.append(val)
        elif isinstance(stmt, ast.AugAssign):
            body_exprs.append(translate_aug_assign(stmt))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            target = None
            val = None
            if isinstance(stmt, ast.AnnAssign):
                target = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                val = translate_expr(stmt.value) if stmt.value else "#f"
            else:
                if len(stmt.targets) == 1:
                    t = stmt.targets[0]
                    if isinstance(t, ast.Name):
                        target = t.id
                        val = translate_expr(stmt.value)
                    elif isinstance(t, ast.Subscript):
                        d = translate_expr(t.value)
                        k = translate_expr(t.slice)
                        v = translate_expr(stmt.value)
                        body_exprs.append(f"(dict-set! {d} {k} {v})")
            if target and val:
                body_exprs.append(f"(set! {target} {val})")
            else:
                body_exprs.append(f";; {ast.dump(stmt)}")
        elif isinstance(stmt, ast.If):
            test = translate_expr(stmt.test)
            then_exprs = translate_block(stmt.body)
            else_body = stmt.orelse
            if else_body:
                else_exprs = translate_block(else_body)
                body_exprs.append(
                    f"(if {test}\n    (begin {' '.join(then_exprs)})\n    (begin {' '.join(else_exprs)}))")
            elif (si + 1 < len(node.body) and isinstance(node.body[si + 1], ast.Return)
                  and len(stmt.body) > 0 and isinstance(stmt.body[-1], ast.Return)):
                next_stmt = node.body[si + 1]
                next_val = translate_expr(next_stmt.value) if next_stmt.value else "#f"
                body_exprs.append(f"(if {test}\n    (begin {' '.join(then_exprs)})\n    {next_val})")
                si += 1
            else:
                body_exprs.append(f"(if {test}\n    (begin {' '.join(then_exprs)}))")
        elif isinstance(stmt, ast.For):
            target = stmt.target.id if isinstance(stmt.target, ast.Name) else "i"
            if isinstance(stmt.iter, ast.Call) and hasattr(stmt.iter.func, 'id') and stmt.iter.func.id == 'range':
                rargs = stmt.iter.args
                if len(rargs) == 1:
                    n = translate_expr(rargs[0])
                    body_parts = translate_block(stmt.body)
                    body_str = " ".join(body_parts)
                    body_exprs.append(
                        f"(do (({target} 0 (+ {target} 1))) ((>= {target} {n})) {body_str})")
                else:
                    body_exprs.append(f";; for range with {len(rargs)} args")
            else:
                body_exprs.append(f";; for loop (untranslated)")
        elif isinstance(stmt, ast.While):
            test = translate_expr(stmt.test)
            body_parts = translate_block(stmt.body)
            body_str = " ".join(body_parts)
            body_exprs.append(f"(let loop () (if {test} (begin {body_str} (loop))))")
        elif isinstance(stmt, ast.Expr):
            body_exprs.append(translate_expr(stmt.value))
        else:
            body_exprs.append(f";; {type(stmt).__name__}: {ast.dump(stmt)}")
        si += 1
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
    for name, info in EXTERN_FUNCTIONS.items():
        lib = info["lib"]
        if lib == "torch_std_helper" and lib not in loaded_libs:
            lines.append('(load-shared-object "/opt/ReScheme/build/libtorch_std_helper.so")')
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
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            output_parts.append(translate_function(node, source_filename))
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
