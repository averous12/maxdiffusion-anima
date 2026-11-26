"""Catch the bug that killed the TPU server on its first request.

`serve()` read `last_gen_id` at line 1504 and assigned it at line 1510. Not being in the
function's `global` list, Python resolved it to a *local* for the whole function, so the read
raised `UnboundLocalError: cannot access local variable 'last_gen_id'` on the very first
request. The server booted perfectly, reported ready, and died the moment anyone used it.

Only statement-level assignment targets are considered, which is exactly the shape that bites:
a plain `name = ...` makes the name local for the entire function, including lines above it.
Loop and comprehension targets are deliberately ignored -- they are bound at their own site and
comparing positions there produces false positives.

Run:  python ops/test_global_completeness.py [path-to-pipeline.py]
"""
import ast
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "src/maxdiffusion/runtime/pipeline.py"


def walk_scope(node):
    """ast.walk without descending into Lambda bodies -- their params are a separate scope,
    and treating them as reads of this one reported a fake bug on `lambda p, ids, mask:`."""
    yield node
    if isinstance(node, ast.Lambda):
        return
    for child in ast.iter_child_nodes(node):
        yield from walk_scope(child)


def own_scope(body):
    """Nodes in this function's scope, excluding nested function/class bodies."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from own_scope(getattr(stmt, field, []) or [])
        for handler in getattr(stmt, "handlers", []) or []:
            yield from own_scope(handler.body)


def main():
    src = open(TARGET, encoding="utf-8").read()
    tree = ast.parse(src)

    problems = []
    checked = 0
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        declared = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Global):
                declared.update(n.names)
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        checked += 1

        assigns = {}      # name -> earliest statement-level assignment position
        reads = []        # (name, position)
        for stmt in own_scope(fn.body):
            # Only a plain name target rebinds. `d[k] = v` / `obj.attr = v` mutate, and the
            # base name in those targets is a Load -- walking into them invented assignments
            # that do not exist and produced false positives.
            targets = []
            # Loop / with / except / walrus targets also bind a name earlier, and missing them
            # produced a false positive on `for index, raw in ...: if not raw[...]`.
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                t = stmt.target
                if isinstance(t, ast.Name):
                    targets.append(t)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    targets.extend(e for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if isinstance(item.optional_vars, ast.Name):
                        targets.append(item.optional_vars)
            elif isinstance(stmt, ast.Try):
                for handler in stmt.handlers:
                    if handler.name:
                        targets.append(ast.Name(id=handler.name, ctx=ast.Store(),
                                                lineno=stmt.lineno, col_offset=0))
            for node in walk_scope(stmt):
                if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                    targets.append(node.target)
                # comprehension targets are their own scope, but treating them as bindings
                # keeps them from being mistaken for reads of a later statement-level assign
                if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                     ast.GeneratorExp)):
                    for gen in node.generators:
                        t = gen.target
                        if isinstance(t, ast.Name):
                            targets.append(t)
                        elif isinstance(t, (ast.Tuple, ast.List)):
                            targets.extend(e for e in t.elts if isinstance(e, ast.Name))
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        targets.extend(e for e in t.elts if isinstance(e, ast.Name))
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(stmt.target, ast.Name):
                    targets.append(stmt.target)
            for n in targets:
                pos = (stmt.lineno, stmt.col_offset)
                assigns[n.id] = min(assigns.get(n.id, pos), pos)
            for node in walk_scope(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    reads.append((node.id, (node.lineno, node.col_offset)))

        first_read = {}
        for name, pos in reads:
            if name not in first_read or pos < first_read[name]:
                first_read[name] = pos

        for name, rpos in sorted(first_read.items()):
            if name in params or name in declared or name not in assigns:
                continue
            if rpos < assigns[name]:     # read happens above the assignment -> unbound local
                problems.append((fn.name, name, rpos[0], assigns[name][0]))

    for fname, name, rline, aline in problems:
        print(f"FAIL {fname}(): '{name}' read at line {rline} but first assigned at line "
              f"{aline}, and not declared global")
    if problems:
        print(f"\nRESULT: FAIL ({len(problems)} read-before-assign across {checked} functions)")
        return 1
    print(f"RESULT: GLOBALS OK ({checked} functions checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
