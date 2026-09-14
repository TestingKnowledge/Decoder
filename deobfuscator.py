"""
deobfuscator.py — Static Lua/Luau deobfuscation engine.

Implements the analysis-rule framework from the build spec:

  RULE 1  Execution entry point (IIFE detection & argument provenance)
  RULE 2  Constant simplification (arithmetic camouflage) via safe folding
  RULE 3  String pool extraction (table constructors, indexed assignments,
          table.insert batches)
  RULE 4  Decoder function recovery + pool substitution into the AST
  RULE 5  VM detection & classification (register / stack, dispatch loop,
          instruction data, opcode families)
  RULE 6  Control-flow flattening detection & state-graph mapping
  RULE 7  Metatable / proxy analysis
  RULE 8  Function wrapping & closure/upvalue analysis
  RULE 9  Arithmetic semantics (Lua-accurate folding incl. floor division,
          32-bit bitwise ops, string coercion)
  RULE 10 String encoding detection (plain / hex / base64 / custom base-N /
          XOR / charcode) with decoder recovery
  RULE 11 Bytecode/VM instruction-table interpretation
  RULE 12 Execution tracing (static CFG walk with provenance)
  RULE 13 Payload recovery with confidence levels — never guesses
  RULE 14 No cross-sample contamination (engine instance is per-run)
  RULE 15+ Hard limits + strict output separation:
          'analysis'      → human report ([UNKNOWN] allowed here ONLY)
          'clean_script'  → pure Lua/Luau source: no comments, no banners,
            no inline [UNKNOWN], nothing before the first line or after
            the last

The engine is fully static: obfuscated code is never executed.
Python's eval() is never used on sample-derived data.
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from lua_base import (
    ENGINE, EngineOverflow, EngineTimeout, LuaTable, LuaValue, Tok, _Limits,
    _quote_lua, lex,
)
from engine_ast import (
    AssignStat, BinOp, BreakStat, Call, CallStat, DoStat, Dot, FalseLit,
    FuncExpr, FuncStat, GenForStat, GotoStat, IfStat, Index, Invoke, LabelStat,
    LocalFuncStat, LocalStat, Name, NilLit, Node, NumForStat, NumLit, Printer,
    RepeatStat, ReturnStat, StrLit, TableLit, TrueLit, UnOp, Vararg, WhileStat,
    normalize_luau_compound, ParseError, Parser,
)

# ══════════════════════════════════════════════════════════════════════════════
# §3  AST UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def walk(node: Node) -> Node:
    """Yield every node in the AST, depth-first, pre-order.

    Iterative (explicit stack) so deeply nested / pathologically long
    obfuscation trees cannot overflow Python's call stack.
    """
    if node is None:
        return
    stack: List[Node] = [node]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        yield node
        if isinstance(node, BinOp):
            stack.append(node.right)
            stack.append(node.left)
        elif isinstance(node, UnOp):
            stack.append(node.operand)
        elif isinstance(node, Index):
            stack.append(node.key)
            stack.append(node.obj)
        elif isinstance(node, Dot):
            stack.append(node.obj)
        elif isinstance(node, Call):
            for a in reversed(node.args):
                stack.append(a)
            stack.append(node.func)
        elif isinstance(node, Invoke):
            for a in reversed(node.args):
                stack.append(a)
            stack.append(node.obj)
        elif isinstance(node, FuncExpr):
            for s in reversed(node.body):
                stack.append(s)
        elif isinstance(node, TableLit):
            for k, v in reversed(node.entries):
                stack.append(v)
                if k is not None:
                    stack.append(k)
        elif isinstance(node, (LocalStat, AssignStat)):
            # visit order (matches original): exprs, then targets
            if isinstance(node, AssignStat):
                for t in reversed(node.targets):
                    stack.append(t)
            for e in reversed(node.exprs):
                stack.append(e)
        elif isinstance(node, CallStat):
            stack.append(node.expr)
        elif isinstance(node, DoStat):
            for s in reversed(node.body):
                stack.append(s)
        elif isinstance(node, WhileStat):
            for s in reversed(node.body):
                stack.append(s)
        elif isinstance(node, RepeatStat):
            stack.append(node.cond)
            for s in reversed(node.body):
                stack.append(s)
        elif isinstance(node, IfStat):
            if node.else_body is not None:
                for s in reversed(node.else_body):
                    stack.append(s)
            for c, b in reversed(node.elseifs):
                for s in reversed(b):
                    stack.append(s)
                stack.append(c)
            for s in reversed(node.then_body):
                stack.append(s)
            stack.append(node.cond)
        elif isinstance(node, NumForStat):
            for s in reversed(node.body):
                stack.append(s)
            if node.step is not None:
                stack.append(node.step)
            stack.append(node.stop)
            stack.append(node.start)
        elif isinstance(node, GenForStat):
            for s in reversed(node.body):
                stack.append(s)
            for e in reversed(node.exprs):
                stack.append(e)
        elif isinstance(node, (FuncStat, LocalFuncStat)):
            # visit order (matches original): body, then name
            if isinstance(node, FuncStat) and node.name is not None:
                stack.append(node.name)
            for s in reversed(node.body):
                stack.append(s)
        elif isinstance(node, ReturnStat):
            for e in reversed(node.exprs):
                stack.append(e)
        # leaves (Name/NumLit/StrLit/Bool/Nil/Vararg/Break/Goto/Label): nothing


def expr_to_source(n: Node) -> str:
    """Render a single expression back to Lua source (for report rows)."""
    p = Printer()
    return p.expr(n).strip()


def is_pure_literal_tree(n: Node) -> bool:
    """True if the expression contains only literals & operators (no names)."""
    for x in walk(n):
        if isinstance(x, (Name, Vararg, Call, Invoke)):
            return False
    return True


# ═════════════════════════════════════════ Table of Contents ═══════════════════
#   §4  CONSTANT FOLDING (RULES 2 & 9)
#   §5  STRING POOLS (RULE 3)
#   §6  DECODERS & ENCODING (RULES 4 & 10)
#   §7  VM & CONTROL FLOW (RULES 5, 6, 11)
#   §8  METATABLES & CLOSURES (RULES 7 & 8)
#   §9  ENTRY & PAYLOAD (RULES 1, 12, 13)
#   §10 REPORT & CLEAN SCRIPT
#   §11 REGEX FALLBACK MODE
#   §12 ENGINE FACADE (DeobfuscationEngine)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# §4  CONSTANT FOLDING (RULES 2 & 9) — arithmetic camouflage removal
# ══════════════════════════════════════════════════════════════════════════════

# Obfuscated samples routinely nest thousands of parens/tables; the parser
# and analyzers use bounded recursion (guard caps + iterative spines), but we
# also lift Python's default limit for the pure-Python call frames involved.
sys.setrecursionlimit(30_000)

_SPINE_FOLD_OPS = {"+", "-", "*", "//", "/", "%", "^", ".."}


def fold_expr(n: Node, lim: _Limits, depth: int = 0) -> Tuple[Optional[LuaValue], Optional[Node]]:
    """Recursively evaluate a literal-only expression.

    Returns (value, replacement_node):
      value            — the folded LuaValue, or None if not fully foldable
      replacement_node — a literal node with the folded value, or None

    Never raises on non-foldable input — returns (None, None) instead.
    A folded LuaValue of kind 'nil' is a REAL nil result; use
    `value is not None` to test foldability, never truthiness.
    """
    if depth > 512:
        return None, None
    lim.tick_step()

    if isinstance(n, NumLit):
        return LuaValue.number(n.value), None
    if isinstance(n, StrLit):
        return LuaValue.string(n.value), None
    if isinstance(n, TrueLit):
        return LuaValue.boolean(True), None
    if isinstance(n, FalseLit):
        return LuaValue.boolean(False), None
    if isinstance(n, NilLit):
        return LuaValue.nil(), None
    if isinstance(n, Vararg):
        return None, None

    if isinstance(n, UnOp):
        v, _ = fold_expr(n.operand, lim, depth + 1)
        if v is None:
            return None, None
        r = v.unop(n.op, lim)
        if r is None:
            return None, None
        return r, _lit_node(r, n.line)

    if isinstance(n, BinOp):
        if n.op in ("and", "or"):
            lv, _ = fold_expr(n.left, lim, depth + 1)
            if lv is None:
                return None, None
            if n.op == "and":
                if not lv.truthy():
                    return lv, _lit_node(lv, n.line)
                rv, _ = fold_expr(n.right, lim, depth + 1)
                if rv is None:
                    return None, None
                return rv, _lit_node(rv, n.line)
            else:
                if lv.truthy():
                    return lv, _lit_node(lv, n.line)
                rv, _ = fold_expr(n.right, lim, depth + 1)
                if rv is None:
                    return None, None
                return rv, _lit_node(rv, n.line)

        if n.op in _SPINE_FOLD_OPS:
            # Iterative chain fold (RULE 2).  Obfuscated scripts pad
            # expressions with thousands of terms; the parse tree is a long
            # chain and recursive descent would blow the depth cap and
            # Python's call stack.  Rather than assuming a fixed lean, we
            # detect the chain direction from the tree itself: the chain
            # continues through whichever child is itself a same-family
            # BinOp.  (Lua: '+'/'-','*','/','//','%' are left-associative;
            # '^' is right-associative; our parser builds '..' chains
            # left-leaning, but explicit parens can produce either shape.)
            # Ambiguous/mixed shapes are refused -- never guess.  Sub-chains
            # reached as items recurse through fold_expr and get their own
            # iterative walk.
            def _is_chain(x: Node) -> bool:
                return isinstance(x, BinOp) and x.op in _SPINE_FOLD_OPS

            left_chains = _is_chain(n.left)
            right_chains = _is_chain(n.right)
            if left_chains and right_chains:
                return None, None        # mixed shape: refuse to guess
            leans_right = right_chains
            items: List[Node] = []
            ops: List[str] = []
            cur: Node = n
            while _is_chain(cur):
                lim.tick_step()
                if leans_right:
                    items.append(cur.left)
                    cur = cur.right
                else:
                    items.append(cur.right)
                    cur = cur.left
            # `ops` must pair with `items` in fold order; rebuild cleanly:
            ops = []
            link = n
            while _is_chain(link):
                ops.append(link.op)
                link = link.right if leans_right else link.left
            # UNIFORM CONCAT FAST PATH (RULE 2/9): a chain of pure string
            # literals joined with '..' is assembled in ONE pass, charging
            # the string budget once (final size) instead of quadratically
            # for every intermediate (which capped chains at a few thousand
            # terms).  Long concat chains are a staple obfuscation pattern.
            if n.op == ".." and all(o == ".." for o in ops) and len(items) >= 4:
                assoc: List[Node] = (items + [cur]) if leans_right \
                    else ([cur] + list(reversed(items)))
                if all(isinstance(x, StrLit) for x in assoc):
                    total = sum(len(x.value) for x in assoc)
                    lim.tick_string(total)
                    joined = "".join(x.value for x in assoc)
                    v = LuaValue.string(joined)
                    return v, _lit_node(v, n.line)
            acc, _ = fold_expr(cur, lim, depth + 1)
            if acc is None:
                return None, None
            if leans_right:
                # right-leaning: fold from the deepest (rightmost) end
                while items:
                    lv, _ = fold_expr(items.pop(), lim, depth + 1)
                    if lv is None:
                        return None, None
                    acc = lv.binop(ops.pop(), acc, lim)
                    if acc is None:
                        return None, None
            else:
                # left-leaning: fold from the leftmost end
                while items:
                    rv, _ = fold_expr(items.pop(), lim, depth + 1)
                    if rv is None:
                        return None, None
                    acc = acc.binop(ops.pop(), rv, lim)
                    if acc is None:
                        return None, None
            return acc, _lit_node(acc, n.line)

        lv, _ = fold_expr(n.left, lim, depth + 1)
        rv, _ = fold_expr(n.right, lim, depth + 1)
        if lv is None or rv is None:
            return None, None
        r = lv.binop(n.op, rv, lim)
        if r is None:
            return None, None
        return r, _lit_node(r, n.line)

    if isinstance(n, TableLit):
        t = LuaTable()
        for k, v in n.entries:
            kv = None
            if k is not None:
                kv, _ = fold_expr(k, lim, depth + 1)
                if kv is None:
                    return None, None
            vv, _ = fold_expr(v, lim, depth + 1)
            if vv is None:
                return None, None
            if kv is None:
                t.list.append(vv)
            else:
                if kv.kind == "num" and kv.int_value() is not None:
                    t.set(kv.int_value(), vv)
                elif kv.kind == "str":
                    t.set(kv.str, vv)
                else:
                    return None, None
        return LuaValue.table(t), None

    if isinstance(n, Index):
        ov, _ = fold_expr(n.obj, lim, depth + 1)
        kv, _ = fold_expr(n.key, lim, depth + 1)
        if ov is not None and ov.kind == "table" and kv is not None:
            key = kv.int_value() if kv.is_int() else (kv.str if kv.kind == "str" else None)
            if key is not None:
                got = ov.tbl.get(key)
                if got is not None:
                    return got, _lit_node(got, n.line)
        return None, None

    return None, None


def _lit_node(v: LuaValue, line: int) -> Optional[Node]:
    """Build a literal AST node from a folded LuaValue (if representable)."""
    if v is None:
        return None
    if v.kind == "num":
        return NumLit(line=line, value=v.num)
    if v.kind == "str":
        return StrLit(line=line, value=v.str)
    if v.kind == "bool":
        return TrueLit(line=line) if v.bval else FalseLit(line=line)
    if v.kind == "nil":
        return NilLit(line=line)
    return None


class ConstFolder:
    """RULE 2 + RULE 9: rewrite the AST in place, folding literal arithmetic.

    Collects a mapping for the report:
        Original expression | Simplified value | Where used | Purpose
    """

    def __init__(self, lim: _Limits):
        self.lim = lim
        self.folded: Dict[str, str] = {}
        self.count = 0
        self.noteworthy: List[str] = []

    def run(self, stats: List[Node]) -> None:
        for s in stats:
            self._stmt(s)

    def _stmt(self, s: Node) -> None:
        self.lim.tick_step()
        if isinstance(s, (LocalStat, AssignStat)):
            s.exprs = [self._expr_maybe_replace(e, s) for e in s.exprs]
            if isinstance(s, AssignStat):
                s.targets = [self._expr_maybe_replace(t, s, is_target=True)
                             for t in s.targets]
        elif isinstance(s, CallStat):
            s.expr = self._expr_maybe_replace(s.expr, s)
        elif isinstance(s, DoStat):
            for x in s.body:
                self._stmt(x)
        elif isinstance(s, WhileStat):
            s.cond = self._expr_maybe_replace(s.cond, s)
            for x in s.body:
                self._stmt(x)
        elif isinstance(s, RepeatStat):
            for x in s.body:
                self._stmt(x)
            s.cond = self._expr_maybe_replace(s.cond, s)
        elif isinstance(s, IfStat):
            s.cond = self._expr_maybe_replace(s.cond, s)
            for x in s.then_body:
                self._stmt(x)
            for idx, (c, b) in enumerate(s.elseifs):
                new_c = self._expr_maybe_replace(c, s)
                if new_c is not c:
                    s.elseifs[idx] = (new_c, b)
            if s.else_body is not None:
                for x in s.else_body:
                    self._stmt(x)
        elif isinstance(s, NumForStat):
            s.start = self._expr_maybe_replace(s.start, s)
            s.stop = self._expr_maybe_replace(s.stop, s)
            if s.step is not None:
                s.step = self._expr_maybe_replace(s.step, s)
            for x in s.body:
                self._stmt(x)
        elif isinstance(s, GenForStat):
            s.exprs = [self._expr_maybe_replace(e, s) for e in s.exprs]
            for x in s.body:
                self._stmt(x)
        elif isinstance(s, (FuncStat, LocalFuncStat, FuncExpr)):
            for x in s.body:
                self._stmt(x)
            if isinstance(s, FuncStat) and s.name is not None:
                self._fold_name(s.name)
        elif isinstance(s, ReturnStat):
            s.exprs = [self._expr_maybe_replace(e, s) for e in s.exprs]
        # Break/Goto/Label — nothing to fold

    def _fold_name(self, n: Node) -> None:
        if isinstance(n, Index):
            n.key = self._expr_maybe_replace(n.key, None)
            self._fold_name(n.obj)

    def _fold_binop_chain(self, e: Node, ctx: Optional[Node]) -> Node:
        """Handle a (possibly enormous) BinOp chain without spine recursion.

        Strategy:
          1. Try folding the WHOLE chain (fold_expr walks chains iteratively
             and refuses ambiguous mixed shapes -- never guesses).
          2. Spine shape (one child a chain, one a small subtree): walk the
             spine iteratively, fully processing each non-chain item via the
             normal expression recursion (bounded subtree depth), then retry
             the whole-chain fold.
          3. Mixed shape (both children chains -- e.g. parenthesized groups):
             recurse ONE level into each child; each child is itself handled
             iteratively here, so stack depth stays O(tree height), and
             parenthesized groups are balanced (log-depth) in practice.

        Returns the (possibly replaced) node.
        """
        def _is_chain(x: Node) -> bool:
            return isinstance(x, BinOp) and x.op in _SPINE_FOLD_OPS

        if not _is_chain(e):
            return e

        # 1) whole-chain fold attempt
        node = self._attempt_whole_fold(e, ctx)
        if node is not None:
            return node

        left_ch = _is_chain(e.left)
        right_ch = _is_chain(e.right)

        # 3) mixed shape: one level of recursion per balanced group
        if left_ch and right_ch:
            e.left = self._expr_maybe_replace(e.left, ctx)
            e.right = self._expr_maybe_replace(e.right, ctx)
            node = self._attempt_whole_fold(e, ctx)
            return node if node is not None else e

        # 2) spine walk (uniform lean): flatten to (nodes, ops) in
        #    association order, process items via bounded recursion, then
        #    iteratively merge adjacent literal pairs.  Merging runs in true
        #    association order (left-assoc ops: left-to-right; right-assoc
        #    shape: right-to-left) so semantics are preserved exactly.
        leans_right = right_ch
        items: List[Node] = []
        chain_ops: List[str] = []
        cur: Node = e
        while _is_chain(cur):
            self.lim.tick_step()
            chain_ops.append(cur.op)
            if leans_right:
                items.append(cur.left)
                cur = cur.right
            else:
                items.append(cur.right)
                cur = cur.left
        head = cur
        # process every non-chain item (subtree recursion, bounded depth)
        for i, item in enumerate(items):
            items[i] = self._expr_maybe_replace(item, ctx)
        # Iterative CUMULATIVE fold along true association order (never
        # folds non-subtree pairs; semantics preserved exactly):
        #   left-lean  : ((((x0 o0 x1) o1 x2) o2 x3) ...) -- accumulator
        #                grows left-to-right; a fold is valid only between
        #                the accumulated literal prefix and the next item.
        #   right-lean : (x0 o0 (x1 o1 (x2 o2 x3))) -- mirrored, suffix
        #                accumulator grows right-to-left.
        if leans_right:
            seq: List[Node] = items + [head]
            ops: List[str] = chain_ops
            acc: Optional[Node] = seq[-1]
            for i in range(len(seq) - 2, -1, -1):
                if isinstance(acc, (NumLit, StrLit, TrueLit, FalseLit, NilLit)) and \
                        self._is_literal(seq[i]):
                    m = self._merge_pair(seq[i], ops[i], acc, ctx)
                    if m is not None:
                        acc = m
                        continue
                acc = BinOp(line=e.line, op=ops[i], left=seq[i], right=acc)
            node = acc
        else:
            seq = [head] + list(reversed(items))
            ops = list(reversed(chain_ops))
            acc = seq[0]
            for i in range(1, len(seq)):
                if isinstance(acc, (NumLit, StrLit, TrueLit, FalseLit, NilLit)) and \
                        self._is_literal(seq[i]):
                    m = self._merge_pair(acc, ops[i - 1], seq[i], ctx)
                    if m is not None:
                        acc = m
                        continue
                acc = BinOp(line=e.line, op=ops[i - 1], left=acc, right=seq[i])
            node = acc
        if node is None:
            return e
        node = self._attempt_whole_fold(node, ctx)
        return node if node is not None else e

    @staticmethod
    def _is_literal(x: Node) -> bool:
        return isinstance(x, (NumLit, StrLit, TrueLit, FalseLit, NilLit))

    def _merge_pair(self, a: Node, op: str, b: Node, ctx: Optional[Node]) -> Optional[Node]:
        """Fold literal `a op b` into a literal node (or None).  Records the fold."""
        synthetic = BinOp(line=a.line, op=op, left=a, right=b)
        try:
            v, _ = fold_expr(synthetic, self.lim)
        except (EngineOverflow, EngineTimeout):
            raise
        if v is None:
            return None
        node = _lit_node(v, a.line)
        if node is None:
            return None
        original = expr_to_source(synthetic)
        simplified = expr_to_source(node)
        if original != simplified and original not in self.folded:
            self.folded[original] = simplified
            self.count += 1
            if len(self.noteworthy) < 40:
                where = f"  (in {type(ctx).__name__} line {ctx.line})" if ctx is not None else ""
                self.noteworthy.append(f"{original} \u2192 {simplified}{where}")
        return node

    def _attempt_whole_fold(self, e: Node, ctx: Optional[Node]) -> Optional[Node]:
        """Fold a pure-literal expression to a literal node; None if not foldable."""
        if not is_pure_literal_tree(e):
            return None
        if isinstance(e, (NumLit, StrLit, TrueLit, FalseLit, NilLit, TableLit)):
            return None
        try:
            v, _ = fold_expr(e, self.lim)
        except (EngineOverflow, EngineTimeout):
            raise
        if v is None:
            return None
        node = _lit_node(v, e.line)
        if node is None:
            return None
        original = expr_to_source(e)
        simplified = expr_to_source(node)
        if original != simplified and original not in self.folded:
            self.folded[original] = simplified
            self.count += 1
            if len(self.noteworthy) < 40:
                where = f"  (in {type(ctx).__name__} line {ctx.line})" if ctx is not None else ""
                self.noteworthy.append(f"{original} \u2192 {simplified}{where}")
        return node

    def _expr_maybe_replace(self, e: Node, ctx: Optional[Node],
                            is_target: bool = False) -> Node:
        if e is None:
            return e
        self.lim.tick_step()

        # BinOp chains (obfuscation padding) are processed ITERATIVELY so a
        # 50k-term chain cannot overflow the call stack.  The chain handler
        # processes every item and attempts the whole-chain fold, so its
        # result is final -- return directly (never re-descend the spine).
        if isinstance(e, BinOp) and e.op in _SPINE_FOLD_OPS:
            return self._fold_binop_chain(e, ctx)

        # recurse into children first (innermost-out folding)
        if isinstance(e, BinOp):
            e.left = self._expr_maybe_replace(e.left, ctx)
            e.right = self._expr_maybe_replace(e.right, ctx)
        elif isinstance(e, UnOp):
            e.operand = self._expr_maybe_replace(e.operand, ctx)
        elif isinstance(e, Index):
            e.obj = self._expr_maybe_replace(e.obj, ctx)
            e.key = self._expr_maybe_replace(e.key, ctx)
        elif isinstance(e, Dot):
            e.obj = self._expr_maybe_replace(e.obj, ctx)
        elif isinstance(e, Call):
            e.func = self._expr_maybe_replace(e.func, ctx)
            e.args = [self._expr_maybe_replace(a, ctx) for a in e.args]
        elif isinstance(e, Invoke):
            e.obj = self._expr_maybe_replace(e.obj, ctx)
            e.args = [self._expr_maybe_replace(a, ctx) for a in e.args]
        elif isinstance(e, TableLit):
            new_entries: List[Tuple[Optional[Node], Node]] = []
            for k, v in e.entries:
                if k is not None:
                    k = self._expr_maybe_replace(k, ctx)
                v = self._expr_maybe_replace(v, ctx)
                new_entries.append((k, v))
            e.entries = new_entries
        elif isinstance(e, FuncExpr):
            for x in e.body:
                self._stmt(x)

        if is_target:
            return e

        if not is_pure_literal_tree(e):
            return e
        if isinstance(e, (NumLit, StrLit, TrueLit, FalseLit, NilLit, TableLit)):
            return e
        try:
            v, _ = fold_expr(e, self.lim)
        except (EngineOverflow, EngineTimeout):
            raise
        if v is None:
            return e
        node = _lit_node(v, e.line)
        if node is None:
            return e
        original = expr_to_source(e)
        simplified = expr_to_source(node)
        if original != simplified and original not in self.folded:
            self.folded[original] = simplified
            self.count += 1
            if len(self.noteworthy) < 40:
                where = f"  (in {type(ctx).__name__} line {ctx.line})" if ctx is not None else ""
                self.noteworthy.append(f"{original} → {simplified}{where}")
        return node


# ═══════════════════════════════════════════════════════════════════════
# §5  STRING POOLS (RULE 3) — obfuscated scripts hide their string
# constants inside big tables and reference them indirectly.  We recover
# the pool statically, resolve aliases, and remember the indexing scheme
# so later phases can substitute pool[<int>] accesses with literals.
# ═══════════════════════════════════════════════════════════════════════

class Pool:
    """A recovered string/constant pool (RULE 3)."""

    def __init__(self, name: str, line: int, origin: str):
        self.name = name
        self.line = line
        self.origin = origin          # "constructor" / "indexed-assign" / "table.insert"
        self.table = LuaTable()
        self.aliases: List[str] = [name]
        self.string_count = 0
        self.all_strings = True

    def note_value(self, v: LuaValue, key: int) -> None:
        if v.kind != "str":
            self.all_strings = False
        else:
            self.string_count += 1
        self.table.set(key, v)

    def get(self, key: int) -> Optional[LuaValue]:
        return self.table.get(key)

    def bounds(self) -> Tuple[Optional[int], Optional[int]]:
        keys = [k for k in list(self.table.hash.keys()) if isinstance(k, int)]
        if not keys and not self.table.list:
            return None, None
        lo, hi = (None, None)
        if keys:
            lo, hi = min(keys), max(keys)
        if self.table.list:
            n = len(self.table.list)
            alo, ahi = 1, 1
            while ahi <= n and self.table.list[ahi - 1] is not None:
                ahi += 1
            alo = 1
            if lo is None:
                lo, hi = alo, ahi - 1
            else:
                lo = min(lo, alo)
                hi = max(hi, ahi - 1)
        return lo, hi

    def size(self) -> int:
        keys = [k for k in list(self.table.hash.keys()) if isinstance(k, int)]
        return max(len(keys), len(self.table.list))


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and float(x).is_integer()


def _int_key(n: NumLit) -> Optional[int]:
    v = n.value
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


class PoolExtractor:
    """RULE 3: find string pools and record how they were built."""

    def __init__(self, lim: "_Limits"):
        self.lim = lim
        self.pools: Dict[str, Pool] = {}
        self.notes: List[str] = []
        self.insert_names = {"table", "table.insert"}
        self.calls_seen = 0
        self.stat_batch: Dict[str, List[int]] = {}

    # -- helpers -----------------------------------------------------------
    def _pool_for_alias(self, name: str) -> Optional[Pool]:
        return self.pools.get(name)

    def _record(self, pool: Pool, key: int, v: LuaValue) -> None:
        if pool.size() >= ENGINE["max_pool_entries"]:
            raise EngineOverflow("string pool too large")
        pool.note_value(v, key)

    # -- constructor pools:  local a = { "x", "y", [5] = "z" } -------------
    def _try_constructor(self, stat: LocalStat) -> None:
        if len(stat.names) != 1 or len(stat.exprs) != 1:
            return
        tname = stat.names[0]
        init = stat.exprs[0]
        if not isinstance(init, TableLit) or len(init.entries) < 3:
            return
        pool = Pool(tname, stat.line, "constructor")
        for k, v in init.entries:
            self.lim.tick_step()
            fold = _pool_literal(v, self.lim)
            if fold is None:
                return                      # not a pure pool — leave untouched
            if k is None:
                key = pool.size() + 1       # array part: 1, 2, 3, ...
            elif isinstance(k, NumLit) and _int_key(k) is not None:
                key = _int_key(k)
            elif isinstance(k, StrLit):
                try:
                    key = int(k.value)
                except ValueError:
                    return                  # non-integer string key — not a pool
            else:
                return
            self._record(pool, key, fold)
        if pool.string_count >= 3:
            self.pools[pool.name] = pool
            self.notes.append(
                f"string pool '{pool.name}' = table constructor at line {pool.line}: "
                f"{pool.size()} entries ({pool.string_count} strings)")

    # -- indexed assignment pools:  a[1] = "x"; a[2] = "y" -----------------
    def _try_indexed_assign(self, stat: AssignStat) -> None:
        for tgt, val in zip(stat.targets, stat.exprs):
            if not (isinstance(tgt, Index) and isinstance(tgt.obj, Name)
                    and isinstance(tgt.key, NumLit) and _int_key(tgt.key) is not None):
                continue
            v = _pool_literal(val, self.lim)
            if v is None:
                continue
            name = tgt.obj.id
            pool = self.pools.get(name)
            if pool is None:
                if name in self.stat_batch and self.stat_batch[name] >= 0:
                    pool = Pool(name, stat.line, "indexed-assign")
                    self.pools[name] = pool
                else:
                    # start a batch from a bare name that is later only used
                    # for reads we can see; be conservative: only names that
                    # already appear as an empty table constructor.
                    continue
            self._record(pool, _int_key(tgt.key), v)
            self.stat_batch[name] = 1

    def _seed_empty_tables(self, stats: List[Node]) -> None:
        """local a = {}  →  remember name as a potential pool container."""
        for s in stats:
            if isinstance(s, LocalStat) and len(s.names) == 1 and len(s.exprs) == 1:
                t, v = s.names[0], s.exprs[0]
                if isinstance(v, TableLit) and not v.entries:
                    self.stat_batch.setdefault(t, 0)

    # -- table.insert batches ----------------------------------------------
    def _try_table_insert(self, stat: CallStat) -> None:
        c = stat.expr
        if not isinstance(c, Call) or not isinstance(c.func, Dot):
            return
        if not (isinstance(c.func.obj, Name) and c.func.obj.id == "table"
                and c.func.name == "insert"):
            return
        args = c.args
        if not args or not isinstance(args[0], Name):
            return
        name = args[0].id
        pool = self.pools.get(name)
        if pool is None:
            if self.stat_batch.get(name) != 0:
                return
            pool = Pool(name, stat.line, "table.insert")
            self.pools[name] = pool
        if len(args) == 2:
            v = _pool_literal(args[1], self.lim)
            if v is None:
                return
            key = pool.size() + 1
            self._record(pool, key, v)
            self.stat_batch[name] = 1
        elif len(args) == 3 and isinstance(args[1], NumLit) and _int_key(args[1]) is not None:
            v = _pool_literal(args[2], self.lim)
            if v is None:
                return
            self._record(pool, _int_key(args[1]), v)
            self.stat_batch[name] = 1
        self.calls_seen += 1

    # -- alias resolution:  local b = a  /  local b = a, c = a -------------
    def _resolve_aliases(self, stats: List[Node]) -> None:
        for _ in range(2):                       # two passes catch chains
            changed = False
            for s in stats:
                if not isinstance(s, LocalStat):
                    continue
                for t, v in zip(s.names, s.exprs):
                    if isinstance(v, Name):
                        p = self.pools.get(v.id)
                        if p is not None and t not in p.aliases:
                            p.aliases.append(t)
                            self.pools[t] = p
                            changed = True
            if not changed:
                break

    def _prune(self) -> None:
        """Drop pools that ended up with too few strings to matter."""
        for name in list(self.pools.keys()):
            p = self.pools[name]
            if p.string_count < 2 and p.size() < 4:
                # keep only if it at least looks like a byte/number pool
                if not (p.all_strings and p.size() >= 2):
                    for a in p.aliases:
                        self.pools.pop(a, None)

    # -- entry ---------------------------------------------------------------
    def run(self, stats: List[Node]) -> None:
        self._seed_empty_tables(stats)
        for s in stats:
            self.lim.tick_step()
            if isinstance(s, LocalStat):
                self._try_constructor(s)
            elif isinstance(s, AssignStat):
                self._try_indexed_assign(s)
            elif isinstance(s, CallStat):
                self._try_table_insert(s)
        self._resolve_aliases(stats)
        self._prune()
        # Report the indexing scheme for each surviving pool (RULE 3).
        for p in {id(p): p for p in self.pools.values()}.values():
            lo, hi = p.bounds()
            if lo is not None:
                base_desc = "0-based" if lo == 0 else ("1-based" if lo == 1 else f"lowest index {lo}")
                self.notes.append(
                    f"pool '{p.name}' indexing: {base_desc}, {p.size()} entries "
                    f"(range {lo}..{hi}), aliases: {', '.join(p.aliases)}")


def _pool_literal(n: Node, lim: "_Limits") -> Optional[LuaValue]:
    """Fold a pure-literal expression to a LuaValue, or None."""
    if not is_pure_literal_tree(n):
        return None
    if isinstance(n, (NumLit, StrLit, TrueLit, FalseLit, NilLit)):
        return _lit_value(n)
    try:
        v, _ = fold_expr(n, lim)
    except (EngineOverflow, EngineTimeout):
        raise
    return v


def _lit_value(n: Node) -> Optional[LuaValue]:
    if isinstance(n, NumLit):
        return LuaValue.number(n.value)
    if isinstance(n, StrLit):
        return LuaValue.string(n.value)
    if isinstance(n, TrueLit):
        return LuaValue.boolean(True)
    if isinstance(n, FalseLit):
        return LuaValue.boolean(False)
    if isinstance(n, NilLit):
        return LuaValue.nil()
    return None


# ═══════════════════════════════════════════════════════════════════════
# §6  DECODERS & ENCODING (RULES 4 & 10) — recover decoder functions,
# classify the encoding of string literals / pools (plain, hex, base64,
# custom base-N, XOR, charcode), decode statically, and substitute the
# decoded literals back into the AST.
# ═══════════════════════════════════════════════════════════════════════

_B64_CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "abcdefghijklmnopqrstuvwxyz"
              "0123456789+/")


# -- encoding classification ------------------------------------------------

def _looks_hex(s: str) -> bool:
    return bool(re.fullmatch(r"(?:[0-9a-fA-F]{2})+", s)) and len(s) >= 4


def _looks_base64(s: str) -> bool:
    """Conservative base64 detection: exact charset plus the mixed-case
    + digit signature of real base64 (stops plain English words like
    'Players' from matching)."""
    if len(s) < 8 or len(s) % 4 == 1:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", s):
        return False
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    return has_upper and has_lower and has_digit


def _looks_baseN(s: str, alphabet: str) -> bool:
    if not alphabet or not s:
        return False
    return all(c in alphabet for c in s)


def classify_encoding(s: str, alphabet: Optional[str] = None) -> str:
    """Heuristic encoding classification (RULE 10)."""
    if not s:
        return "empty"
    if alphabet and len(alphabet) >= 8 and _looks_baseN(s, alphabet):
        return "custom base-N"
    if _looks_hex(s):
        return "hex"
    if _looks_base64(s):
        return "base64"
    if re.fullmatch(r"(?:\d+\s*)+", s) and len(s.split()) > 2:
        return "charcode"
    return "plain"


# -- static decoders (never execute sample code) ---------------------------

def _decode_hex(s: str) -> bytes:
    return bytes.fromhex(s)


def _decode_base64(s: str) -> bytes:
    pad = (-len(s)) % 4
    raw = s + "=" * pad
    import base64 as _b64
    return _b64.b64decode(raw, validate=False)


def _decode_custom_baseN(s: str, alphabet: str) -> int:
    """Big integer in a custom base (alphabet order defines digit value)."""
    base = len(alphabet)
    acc = 0
    for ch in s:
        acc = acc * base + alphabet.index(ch)
        if acc.bit_length() > 64_000_000:
            raise EngineOverflow("custom base-N decode too large")
    return acc


def _decode_charcode(nums: List[int]) -> bytes:
    return bytes(_to_byte(n) for n in nums)


def _to_byte(n: int) -> int:
    return int(n) & 0xFF


def _apply_xor(data: bytes, key: bytes) -> bytes:
    if not key:
        return data
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % len(key)]
    return bytes(out)


def _bytes_to_display(data: bytes, limit: int = 400) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


# -- decoder-function recovery ----------------------------------------------
#
# Obfuscated scripts define small decoder helpers, e.g.
#
#   local function d(i) return string.char(i % 256) end
#   local function x(s, k) ... bit32.bxor(string.byte(s, i), k) ... end
#
# We recognize the common shapes and record them as evidence instead of
# guessing what they do.

_DECODER_PATTERNS: List[Tuple[str, str]] = [
    ("string.char", "charcode assembler"),
    ("string.byte", "byte extraction"),
    ("bit32.bxor", "XOR layer"),
    ("bit.bxor", "XOR layer"),
    ("bnot", "bitwise NOT layer"),
    ("char", "charcode assembler"),
    ("gsub", "string rewrite layer"),
    ("base64", "base64 layer"),
    ("from_base64", "base64 layer"),
]


class DecoderInfo:
    def __init__(self, name: str, line: int, kind: str):
        self.name = name
        self.line = line
        self.kind = kind          # human description
        self.calls: int = 0
        self.decoded_samples: List[str] = []

    def describe(self) -> str:
        return f"decoder '{self.name}' at line {self.line} ({self.kind}), used {self.calls}×"


def _identify_decoder(func: FuncExpr, name: str) -> Optional[str]:
    """Classify a function by the globals it references."""
    funcs: List[str] = []
    for n in walk(func):
        if isinstance(n, Dot):
            path = _dot_path(n)
            if path:
                funcs.append(path)
    joined = " | ".join(sorted(set(funcs))) or "(no globals)"
    for pat, kind in _DECODER_PATTERNS:
        if any(pat in f for f in funcs):
            return f"{kind} [{joined}]"
    return None


def _dot_path(n: Node) -> Optional[str]:
    """Render a Name/Dot/Index chain like 'string.char' or 'a.b.c'."""
    parts: List[str] = []
    cur: Node = n
    while True:
        if isinstance(cur, Dot):
            parts.append(cur.name)
            cur = cur.obj
        elif isinstance(cur, Name):
            parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        else:
            return None


class DecoderAnalyzer:
    """RULES 4 & 10: find decoder helpers and decode what we can."""

    def __init__(self, lim: "_Limits", pools: Dict[str, Pool]):
        self.lim = lim
        self.pools = pools
        self.decoders: Dict[str, DecoderInfo] = {}
        self.encoding_notes: List[str] = []
        self.decoded_literals: Dict[str, str] = {}   # original → decoded text
        self.xor_keys: List[str] = []

    def _register(self, name: str, line: int, func: FuncExpr) -> None:
        if name in self.decoders:
            return
        kind = _identify_decoder(func, name)
        if kind is None:
            return
        self.decoders[name] = DecoderInfo(name, line, kind)

    def _collect_local_funcs(self, stats: List[Node]) -> None:
        for s in stats:
            if isinstance(s, LocalFuncStat):
                self._register(s.name, s.line, s)          # inline body
            elif isinstance(s, LocalStat):
                for t, v in zip(s.names, s.exprs):
                    if isinstance(v, FuncExpr):
                        self._register(t, s.line, v)
            elif isinstance(s, AssignStat):
                for t, v in zip(s.targets, s.exprs):
                    if isinstance(t, Name) and isinstance(v, FuncExpr):
                        self._register(t.id, s.line, v)

    def _count_decoder_calls(self, stats: List[Node]) -> None:
        for s in stats:
            for n in walk(s):
                if isinstance(n, Call) and isinstance(n.func, Name):
                    d = self.decoders.get(n.func.id)
                    if d is not None:
                        d.calls += 1

    # -- literal encoding analysis -----------------------------------------
    def _analyze_literal_encodings(self, stats: List[Node]) -> None:
        seen: set = set()
        for s in stats:
            for n in walk(s):
                if not isinstance(n, StrLit):
                    continue
                self.lim.tick_step()
                val = n.value
                if val in seen or len(val) < 4 or len(val) > 8192:
                    continue
                seen.add(val)
                enc = classify_encoding(val)
                if enc == "plain":
                    continue
                note = f"string literal encoded as {enc} (len {len(val)})"
                decoded = self._try_decode(val)
                if decoded is not None:
                    note += f" → statically decoded: {_bytes_to_display(decoded, 120)}"
                self.encoding_notes.append(note)
                if len(self.encoding_notes) > 40:
                    return

    def _try_decode(self, val: str) -> Optional[bytes]:
        enc = classify_encoding(val)
        try:
            if enc == "hex":
                return _decode_hex(val)
            if enc == "base64" and _looks_base64(val):
                return _decode_base64(val)
        except (ValueError, EngineOverflow):
            return None
        return None

    # -- XOR key discovery: local k = "..." used with bxor -----------------
    def _find_xor_keys(self, stats: List[Node]) -> None:
        candidates: Dict[str, str] = {}
        for s in stats:
            for n in walk(s):
                if isinstance(n, LocalStat):
                    for t, v in zip(n.names, n.exprs):
                        if isinstance(v, StrLit) and 1 <= len(v.value) <= 64:
                            candidates.setdefault(t, v.value)
        for s in stats:
            hit = False
            for n in walk(s):
                if isinstance(n, Dot) and _dot_path(n) in ("bit32.bxor", "bit.bxor"):
                    hit = True
                    break
            if hit:
                break
        self.xor_keys = list(candidates.values())[:4]

    # -- pool decoding -------------------------------------------------------
    def _decode_pool_strings(self) -> None:
        for pool in {id(p): p for p in self.pools.values()}.values():
            lo, hi = pool.bounds()
            if lo is None:
                continue
            encs: Dict[str, int] = {}
            sample_decoded = 0
            for k in range(min(lo, 0), hi + 1):
                v = pool.get(k)
                if v is None or v.kind != "str":
                    continue
                self.lim.tick_step()
                enc = classify_encoding(v.str)
                if enc == "plain":
                    continue
                encs[enc] = encs.get(enc, 0) + 1
                if sample_decoded < 3:
                    dec = self._try_decode(v.str)
                    if dec is not None:
                        self.encoding_notes.append(
                            f"pool '{pool.name}'[{k}] is {enc} → "
                            f"{_bytes_to_display(dec, 100)!r}")
                        sample_decoded += 1
            if encs:
                dominant = max(encs, key=lambda e: encs[e])
                self.encoding_notes.append(
                    f"pool '{pool.name}' strings appear {dominant} "
                    f"({sum(encs.values())} of {pool.string_count} strings)")

    # -- entry -----------------------------------------------------------------
    def run(self, stats: List[Node]) -> None:
        self._collect_local_funcs(stats)
        self._count_decoder_calls(stats)
        self._analyze_literal_encodings(stats)
        self._find_xor_keys(stats)
        self._decode_pool_strings()


# -- pool substitution into the AST (RULE 4) --------------------------------

class PoolSubstitutor:
    """Rewrite pool[i] reads into their literal values.

    Only indices we have actually recovered from the source are replaced;
    anything else is left untouched (never guess — RULE 13/15).
    """

    def __init__(self, lim: "_Limits", pools: Dict[str, Pool]):
        self.lim = lim
        self.pools = pools
        self.substitutions = 0
        self.notes: List[str] = []

    def _resolve_name(self, n: Node) -> Optional[Pool]:
        if isinstance(n, Name):
            return self.pools.get(n.id)
        return None

    def run(self, stats: List[Node]) -> None:
        for s in stats:
            self._stmt(s)
        if self.substitutions:
            self.notes.append(
                f"substituted {self.substitutions} pool reference(s) with recovered "
                f"literal values")

    def _stmt(self, s: Node) -> None:
        self.lim.tick_step()
        # generic child traversal
        for field, children in _children_of(s):
            for c in children:
                if field == "targets":
                    continue                # never rewrite assignment targets
                self._dispatch(c, s)

    def _dispatch(self, n: Node, parent: Node) -> None:
        if isinstance(n, Index) and isinstance(n.obj, Name):
            pool = self.pools.get(n.obj.id)
            if pool is not None and isinstance(n.key, NumLit):
                key = _int_key(n.key)
                if key is not None:
                    v = pool.get(key)
                    if v is not None:
                        node = _lit_node(v, n.line)
                        if node is not None:
                            self._replace_in_parent(n, node, parent)
                            self.substitutions += 1
                            return
        for field, children in _children_of(n):
            for c in children:
                if isinstance(c, Node):
                    self._dispatch(c, n)
                elif isinstance(c, (list, tuple)):
                    for cc in c:
                        if isinstance(cc, Node):
                            self._dispatch(cc, n)

    def _replace_in_parent(self, old: Node, new: Node, parent: Node) -> None:
        """Replace old with new wherever it appears directly under parent."""
        for field in _fields_of(parent):
            cur = getattr(parent, field, None)
            if cur is old:
                setattr(parent, field, new)
                return
            if isinstance(cur, list):
                for i, item in enumerate(cur):
                    if item is old:
                        cur[i] = new
                        return
                    if isinstance(item, tuple) and len(item) == 2:
                        k, v = item
                        if v is old:
                            cur[i] = (k, new)
                            return
                        if k is old:
                            cur[i] = (new, v)
                            return


def _noop(*a) -> None:
    pass


_CHILD_FIELDS: Dict[str, Any] = {}


def _fields_of(node: Node) -> List[str]:
    return [f for f in vars(node).keys()]


def _children_of(node: Node) -> List[Tuple[str, List[Node]]]:
    """Return (field, [child nodes]) pairs for generic traversal.

    Handles plain Node fields, List[Node] fields, and pair-list fields
    (TableLit.entries, IfStat.elseifs) by flattening their Node parts.
    """
    out: List[Tuple[str, List[Node]]] = []
    for field, val in vars(node).items():
        if isinstance(val, Node):
            out.append((field, [val]))
        elif isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, Node):
                out.append((field, list(val)))
            elif isinstance(first, tuple):
                flat: List[Node] = []
                for item in val:
                    if isinstance(item, tuple):
                        for part in item:
                            if isinstance(part, Node):
                                flat.append(part)
                            elif isinstance(part, list):
                                flat.extend(x for x in part if isinstance(x, Node))
                    elif isinstance(item, Node):
                        flat.append(item)
                if flat:
                    out.append((field, flat))
    return out


# ═══════════════════════════════════════════════════════════════════════
# §7  VM & CONTROL FLOW (RULES 5, 6, 11) — detect virtual-machine based
# obfuscation (dispatch loop, instruction table, register/stack design),
# control-flow flattening state graphs, and interpret simple bytecode
# instruction tables statically (never executing the sample).
# ═══════════════════════════════════════════════════════════════════════

_VM_EVIDENCE_WEIGHT: Dict[str, int] = {
    "while true do dispatch": 5,
    "while 1 do dispatch": 5,
        "local pc = 0 while loop": 4,
    "instruction table": 4,
    "string.byte probe": 3,
    "string.char assembly": 3,
    "bitwise op family": 3,
    "arith decode family": 3,
    "loadstring/load": 3,
    "loadstring/load with string arg": 3,
    "vmvar shorthand names": 2,
    "lshift/rshift family": 2,
    "lshift family": 2,
    "rshift family": 2,
    "monotone index var": 1,
}


def _body_source(stats: List[Node], limit: int = 4) -> str:
    p = Printer()
    for s in stats:
        p.emit_stmt(s)
    return p.result()[:limit * 4000]


def _dotpaths(stats: List[Node], lim: "_Limits") -> List[str]:
    out: List[str] = []
    for s in stats:
        for n in walk(s):
            lim.tick_step()
            if isinstance(n, Dot):
                p = _dot_path(n)
                if p and p not in out:
                    out.append(p)
    return out


class VMInfo:
    """What we learned about a detected virtual machine (RULE 5)."""

    def __init__(self) -> None:
        self.detected = False
        self.design: str = "unknown"          # register / stack / unknown
        self.dispatch_kind: str = "unknown"   # while-true / repeat / for / goto
        self.dispatch_var: Optional[str] = None
        self.instruction_table: Optional[str] = None
        self.opcode_count: Optional[int] = None
        self.opcode_families: List[str] = []
        self.evidence: List[str] = []
        self.score = 0


def _detect_dispatch_loop(stats: List[Node], lim: "_Limits") -> Optional[Tuple[str, Optional[str]]]:
    """Find a `while true do ... end` / `repeat ... until` dispatcher."""
    best: Optional[Tuple[str, Optional[str]]] = None
    for s in stats:
        for n in walk(s):
            lim.tick_step()
            if isinstance(n, WhileStat):
                c = n.cond
                if isinstance(c, TrueLit) or (isinstance(c, NumLit) and c.value == 1):
                    var = None
                    for inner in walk(n):
                        if isinstance(inner, NumForStat) and inner.var:
                            var = inner.var
                            break
                    if best is None:
                        best = ("while true", var)
            elif isinstance(n, RepeatStat) and isinstance(n.cond, TrueLit):
                if best is None:
                    best = ("repeat-until", None)
    return best


def _find_instruction_tables(stats: List[Node], lim: "_Limits") -> List[Tuple[str, int, TableLit]]:
    """Instruction tables: big numeric tables assigned to a single local."""
    out: List[Tuple[str, int, TableLit]] = []
    for s in stats:
        if isinstance(s, LocalStat) and len(s.names) == 1 and len(s.exprs) == 1:
            t, v = s.names[0], s.exprs[0]
            if isinstance(v, TableLit) and len(v.entries) >= 8:
                nums = 0
                for k, val in v.entries:
                    lim.tick_step()
                    if isinstance(val, NumLit):
                        nums += 1
                if nums >= 6:
                    out.append((t, s.line, v))
    return out


def _opcode_families(dotpaths: List[str]) -> List[str]:
    fams: List[str] = []
    for p in dotpaths:
        if "lshift" in p or "rshift" in p:
            if "shift family" not in fams:
                fams.append("shift family")
        if "bxor" in p or "band" in p or "bor" in p:
            if "bitwise family" not in fams:
                fams.append("bitwise family")
        if "byte" in p:
            if "byte fetch" not in fams:
                fams.append("byte fetch")
        if "char" in p:
            if "char assembly" not in fams:
                fams.append("char assembly")
        if p.endswith(".sub"):
            if "string sub" not in fams:
                fams.append("string sub")
    return fams


def _vm_style_names(stats: List[Node], lim: "_Limits") -> List[str]:
    """Short/obfuscated names typical of VM-generated code."""
    pat = re.compile(
        r"^(?:v[0-9]*|vm[A-Z]?[0-9]*|op[c]?[0-9]*|inst[0-9]*|pc|c[0-9]+|d[0-9]+|e[0-9]+|f[0-9]+)$")
    out: List[str] = []
    seen: set = set()
    for s in stats:
        for n in walk(s):
            lim.tick_step()
            if isinstance(n, Name) and pat.match(n.id) and n.id not in seen:
                seen.add(n.id)
                out.append(n.id)
    return out


class VMAnalyzer:
    """RULE 5: detect & classify the VM."""

    def __init__(self, lim: "_Limits"):
        self.lim = lim
        self.info = VMInfo()
        self.notes: List[str] = []

    def run(self, stats: List[Node]) -> VMInfo:
        dotpaths = _dotpaths(stats, self.lim)
        tables = _find_instruction_tables(stats, self.lim)
        dispatch = _detect_dispatch_loop(stats, self.lim)

        if dispatch is not None:
            kind, var = dispatch
            self.info.dispatch_kind = kind
            self.info.dispatch_var = var
            self.info.score += 3
            self.info.evidence.append(
                f"dispatch loop detected: `{kind} do`"
                + (f", state/index variable `{var}`" if var else ""))

        if tables:
            tname, tline, tnode = tables[0]
            self.info.instruction_table = tname
            self.info.opcode_count = len(tnode.entries)
            self.info.score += 4
            self.info.evidence.append(
                f"instruction table '{tname}' at line {tline} with "
                f"{len(tnode.entries)} entries")
            if len(tables) > 1:
                self.info.evidence.append(
                    f"{len(tables)} large numeric tables found "
                    f"({', '.join(t[0] for t in tables[:4])})")
            self._classify_design(stats, tname)

        fams = _opcode_families(dotpaths)
        self.info.opcode_families = fams
        for f in fams:
            self.info.score += 1
            self.info.evidence.append(f"opcode family evidence: {f}")

        for p in dotpaths:
            if p in ("loadstring", "load") or p.startswith("loadstring."):
                self.info.score += 3
                self.info.evidence.append("loadstring/load call site (dynamic chunk loading)")
                break

        vm_names = _vm_style_names(stats, self.lim)
        if vm_names:
            self.info.score += min(2, len(vm_names) // 2)
            self.info.evidence.append("VM-style shorthand locals: " + ", ".join(vm_names[:8]))

        if self.info.design == "unknown" and tables:
            self._classify_design(stats, tables[0][0])

        self.info.detected = self.info.score >= 5
        if self.info.detected:
            self.notes.append(
                f"virtual machine detected (score {self.info.score}) — "
                f"{self.info.design} design, {self.info.dispatch_kind} dispatch")
            for e in self.info.evidence[:6]:
                self.notes.append(f"  • {e}")
        elif self.info.score >= 2:
            self.notes.append(
                f"weak VM-like structure (score {self.info.score}); "
                "not conclusive without a dispatch loop + instruction data")
        return self.info

    def _classify_design(self, stats: List[Node], tname: str) -> None:
        """Register VM: heavy `t[i]` indexed reads inside the loop;
        Stack VM: numeric `pc` incremented + table.insert stack ops."""
        reg = 0
        stack = 0
        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if isinstance(n, Index) and isinstance(n.obj, Name) and n.obj.id == tname:
                    reg += 1
                if isinstance(n, Call) and isinstance(n.func, Dot) \
                        and _dot_path(n.func) == "table.insert":
                    stack += 1
        source = _body_source(stats, 1)
        dvar = self.info.dispatch_var
        if re.search(r"\bpc\s*=\s*pc\s*\+\s*1", source) or (
                dvar and re.search(
                    rf"\b{re.escape(dvar)}\s*=\s*{re.escape(dvar)}\s*\+\s*1", source)):
            stack += 3
        if reg >= 4 and stack == 0:
            self.info.design = "register"
        elif stack >= 2:
            self.info.design = "stack"
        elif reg >= 2:
            self.info.design = "register"


# -- control-flow flattening (RULE 6) ---------------------------------------

class FlatteningInfo:
    def __init__(self) -> None:
        self.detected = False
        self.state_var: Optional[str] = None
        self.states: int = 0
        self.evidence: List[str] = []
        self.state_actions: List[str] = []


class FlatteningAnalyzer:
    """RULE 6: control-flow flattening — a state variable drives a giant
    `while true do if state == N then ... state = M end end` chain."""

    def __init__(self, lim: "_Limits"):
        self.lim = lim
        self.info = FlatteningInfo()
        self.notes: List[str] = []

    def run(self, stats: List[Node]) -> FlatteningInfo:
        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if not isinstance(n, WhileStat):
                    continue
                if not (isinstance(n.cond, TrueLit) or
                        (isinstance(n.cond, NumLit) and n.cond.value == 1)):
                    continue
                self._examine_loop(n)
                if self.info.detected:
                    break
            if self.info.detected:
                break
        if self.info.detected:
            self.notes.append(
                f"control-flow flattening detected: state machine on "
                f"`{self.info.state_var}` with {self.info.states} states")
            for e in self.info.evidence[:5]:
                self.notes.append(f"  • {e}")
            for a in self.info.state_actions[:8]:
                self.notes.append(f"  • state {a}")
        return self.info

    def _examine_loop(self, loop: WhileStat) -> None:
        arms = 0
        state_var: Optional[str] = None
        actions: List[str] = []
        for s in loop.body:
            if isinstance(s, IfStat):
                r = self._analyze_ifchain(s)
                if r is not None:
                    sv, count, acts = r
                    arms += count
                    if state_var is None:
                        state_var = sv
                    if sv == state_var:
                        actions.extend(acts[:6])
        if arms >= 3 and state_var is not None:
            self.info.detected = True
            self.info.state_var = state_var
            self.info.states = arms
            self.info.evidence.append(
                f"`while true do` wrapper around a {arms}-arm state dispatch on "
                f"the same variable")
            nexts = self._next_states(loop)
            if nexts:
                self.info.evidence.append(
                    f"state transitions: {', '.join(nexts[:10])}")
            self.info.state_actions = actions

    def _analyze_ifchain(self, ifs: IfStat) -> Optional[Tuple[str, int, List[str]]]:
        """Count `if v == k` arms across if/elseif chain; return
        (var, arms, per-arm summaries)."""
        arms = 0
        var: Optional[str] = None
        actions: List[str] = []
        pairs: List[Tuple[Node, List[Node]]] = [(ifs.cond, ifs.then_body)]
        pairs.extend(ifs.elseifs)
        for cond, body in pairs:
            m: Optional[str] = None
            if isinstance(cond, BinOp) and cond.op == "==":
                if isinstance(cond.left, Name):
                    var = cond.left.id
                    m = _const_of(cond.right)
                elif isinstance(cond.right, Name):
                    var = cond.right.id
                    m = _const_of(cond.left)
            if m is None and var is not None:
                m = "?"
            if m is None:
                break
            arms += 1
            acts = _summarize_block(body)
            actions.append(f"{m}: {acts}")
        if arms >= 2 and var is not None:
            return var, arms, actions
        return None

    def _next_states(self, loop: WhileStat) -> List[str]:
        out: List[str] = []
        for s in walk(loop):
            if isinstance(s, AssignStat):
                for t, v in zip(s.targets, s.exprs):
                    if isinstance(t, Name) and isinstance(v, NumLit):
                        out.append(f"{t.id}={_num_repr(v.value)}")
                    elif isinstance(t, Name) and isinstance(v, BinOp) and v.op == "+":
                        l1 = _const_of(v.left)
                        r1 = _const_of(v.right)
                        if l1 == "1" and isinstance(v.right, Name):
                            out.append(f"{t.id}={v.right.id}+1")
                        elif r1 == "1" and isinstance(v.left, Name):
                            out.append(f"{t.id}={v.left.id}+1")
        return out


def _const_of(n: Node) -> Optional[str]:
    if isinstance(n, NumLit):
        return str(_num_repr(n.value))
    if isinstance(n, StrLit):
        return n.value
    return None


def _num_repr(v: Any) -> Any:
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _summarize_block(body: List[Node]) -> str:
    """One-line human summary of a flattened state's action."""
    for s in body:
        if isinstance(s, CallStat) and isinstance(s.expr, Call):
            f = s.expr.func
            if isinstance(f, Name):
                args = ", ".join(_short(a) for a in s.expr.args[:3])
                return f"{f.id}({args})"
            if isinstance(f, Dot):
                p = _dot_path(f)
                if p:
                    args = ", ".join(_short(a) for a in s.expr.args[:3])
                    return f"{p}({args})"
        if isinstance(s, ReturnStat):
            return "return"
        if isinstance(s, LocalStat) and s.names:
            return f"{s.names[0]} = …"
        if isinstance(s, AssignStat) and s.targets and isinstance(s.targets[0], Name):
            return f"{s.targets[0].id} = …"
    return "…"


def _short(n: Node) -> str:
    if isinstance(n, StrLit):
        v = n.value
        return repr(v[:24] + "…" if len(v) > 24 else v)
    if isinstance(n, NumLit):
        return str(_num_repr(n.value))
    if isinstance(n, Name):
        return n.id
    if isinstance(n, Call) and isinstance(n, FuncExpr) is False and isinstance(n.func, Name):
        return f"{n.func.id}(…)"
    return "…"


# ═══════════════════════════════════════════════════════════════════════
# §8  METATABLES & CLOSURES (RULES 7 & 8) — metatable/proxy defense
# structures, function wrapping chains, and captured upvalues.
# ═══════════════════════════════════════════════════════════════════════

class MetaInfo:
    def __init__(self) -> None:
        self.events: List[str] = []          # e.g. "__index", "__call"
        self.set_calls: int = 0
        self.defense_kind: str = ""          # e.g. "tamper wall", "self-destruct"
        self.evidence: List[str] = []


class ClosureInfo:
    def __init__(self) -> None:
        self.nested_depth_max: int = 0
        self.wrapper_chains: List[str] = []
        self.upvalues: List[str] = []
        self.evidence: List[str] = []


class MetaAnalyzer:
    """RULE 7: metatables & proxies."""

    def __init__(self, lim: "_Limits"):
        self.lim = lim
        self.info = MetaInfo()
        self.notes: List[str] = []

    def run(self, stats: List[Node]) -> MetaInfo:
        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if isinstance(n, Call):
                    name = None
                    if isinstance(n.func, Name):
                        name = n.func.id
                    elif isinstance(n.func, Dot):
                        name = _dot_path(n.func)
                    if name in ("setmetatable", "setmetainfo") or (
                            name and name.endswith(".setmetatable")):
                        self.info.set_calls += 1
                        ev = self._events_from(n.args)
                        if ev:
                            self.info.events.extend(ev)
        if self.info.set_calls:
            self.notes.append(
                f"{self.info.set_calls} setmetatable call(s); "
                f"metamethods: {', '.join(sorted(set(self.info.events))[:8]) or 'none found'}")
            for e in self.info.evidence[:6]:
                self.notes.append(f"  • {e}")
        return self.info

    def _events_from(self, args: List[Node]) -> List[str]:
        out: List[str] = []
        for a in args:
            if not isinstance(a, TableLit):
                continue
            for k, _v in a.entries:
                if isinstance(k, StrLit) and k.value.startswith("__"):
                    out.append(k.value)
                    self.info.evidence.append(f"metamethod {k.value} defined")
        return out


class ClosureAnalyzer:
    """RULE 8: function wrapping & closures."""

    def __init__(self, lim: "_Limits"):
        self.lim = lim
        self.info = ClosureInfo()
        self.notes: List[str] = []

    def run(self, stats: List[Node]) -> ClosureInfo:
        def depth_funcs(f: FuncExpr, d: int) -> int:
            best = d
            for s in f.body:
                for n in walk(s):
                    if isinstance(n, FuncExpr):
                        best = max(best, depth_funcs(n, d + 1))
            return best

        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if isinstance(n, FuncExpr):
                    self.info.nested_depth_max = max(self.info.nested_depth_max,
                                                     depth_funcs(n, 1))
        chains = self._wrapper_chains(stats)
        self.info.wrapper_chains = chains
        upv = self._upvalues(stats)
        self.info.upvalues = upv
        if self.info.nested_depth_max >= 2:
            self.notes.append(
                f"functions nested {self.info.nested_depth_max} deep "
                "(wrapper/proxy indirection likely)")
        if chains:
            self.notes.append(f"function alias chain: {' → '.join(chains)}")
        if upv:
            self.notes.append(f"captured upvalue names: {', '.join(upv[:10])}")
        return self.info

    def _wrapper_chains(self, stats: List[Node]) -> List[str]:
        """local a = <fn>; local b = a; ... — a chain of renames."""
        renames: Dict[str, str] = {}
        for s in stats:
            if isinstance(s, LocalStat):
                for t, v in zip(s.names, s.exprs):
                    if isinstance(v, Name):
                        renames[t] = v.id
        best: List[str] = []
        for start in renames:
            if start in renames.values():
                continue
            chain = [start]
            cur = start
            for _ in range(8):
                nxt = renames.get(cur)
                if nxt is None:
                    break
                chain.append(nxt)
                cur = nxt
            if len(chain) > len(best):
                best = chain
        return best

    def _upvalues(self, stats: List[Node]) -> List[str]:
        """Names referenced inside nested functions that are not declared
        in any enclosing visible scope — captured locals (upvalues)."""
        declared: set = set()
        referenced: set = set()
        # everything declared anywhere (conservative superset)
        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if isinstance(n, LocalStat):
                    declared.update(n.names)
                elif isinstance(n, LocalFuncStat):
                    declared.add(n.name)
                elif isinstance(n, FuncStat):
                    declared.update(_funcname_parts(n))
                elif isinstance(n, FuncExpr):
                    declared.update(n.params)
                elif isinstance(n, Name):
                    referenced.add(n.id)
        captured = referenced - declared
        out = [n for n in sorted(captured) if not _is_known_global(n)]
        return out[:20]


def _funcname_parts(n: FuncStat) -> List[str]:
    parts: List[str] = []
    cur: Node = n.name
    stack: List[str] = []
    while True:
        if isinstance(cur, Dot):
            stack.append(cur.name)
            cur = cur.obj
        elif isinstance(cur, Name):
            stack.append(cur.id)
            break
        else:
            break
    parts.extend(reversed(stack))
    return parts


_KNOWN_GLOBALS: set = set(
    "print write warn type tostring tonumber pairs ipairs select error assert pcall "
    "xpcall setmetatable getmetatable require rawget rawset rawequal rawlen unpack "
    "next collectgarbage dofile loadstring load string table math bit bit32 os io "
    "coroutine game workspace players shared script Instance typeof newproxy "
    "setfenv getfenv Vector3 Color3 CFrame tick wait spawn delay task Enum UDim "
    "UDim2 Random Ray".split()
)


def _is_known_global(name: str) -> bool:
    return name in _KNOWN_GLOBALS or name.startswith(
        ("getgenv", "hookmetamethod", "hookfunction", "getrawmetatable"))


# ═══════════════════════════════════════════════════════════════════════
# §9  ENTRY & PAYLOAD (RULES 1, 12, 13) — find the execution entry
# point (IIFE), trace execution statically, and recover the payload with
# a provenance chain and confidence level.  We NEVER guess: anything we
# cannot derive is marked [UNKNOWN] in the *analysis report only*.
# ═══════════════════════════════════════════════════════════════════════

class EntryInfo:
    def __init__(self) -> None:
        self.found = False
        self.kind: str = ""                  # "IIFE" / "IIFE (method)" / "direct call"
        self.line: Optional[int] = None
        self.params: List[str] = []
        self.args: List[Optional[str]] = []  # provenance per argument
        self.body_statements: int = 0
        self.callee_globals: List[str] = []  # globals the entry body touches


class PayloadInfo:
    def __init__(self) -> None:
        self.recovered = False
        self.confidence = ""                 # high / medium / low
        self.provenance: List[str] = []      # chain of transformations
        self.calls_of_interest: List[str] = []
        self.loadstring_targets: List[str] = []
        self.unknowns: List[str] = []
        self.summary: str = ""


class EntryAnalyzer:
    """RULE 1 + RULE 12: entry point detection & static trace."""

    def __init__(self, lim: "_Limits"):
        self.lim = lim
        self.info = EntryInfo()
        self.notes: List[str] = []

    def run(self, stats: List[Node]) -> EntryInfo:
        # 1) classic IIFE:  (function(...) ... end)(args)
        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if isinstance(n, Call) and isinstance(n.func, FuncExpr):
                    self.info.found = True
                    self.info.kind = "IIFE"
                    self.info.line = n.func.line
                    self.info.params = list(n.func.params)
                    self.info.args = [self._arg_origin(a) for a in n.args]
                    self.info.body_statements = len(n.func.body)
                    self.info.callee_globals = _globals_touched(n.func.body, self.lim)
                    if n.func.is_vararg:
                        self.info.kind = "IIFE (vararg)"
                    break
            if self.info.found:
                break
        # 2) fallback: function defined then immediately called once
        if not self.info.found:
            for s in stats:
                if isinstance(s, CallStat) and isinstance(s.expr, Call) \
                        and isinstance(s.expr.func, Name):
                    nm = s.expr.func.id
                    for other in stats:
                        if isinstance(other, LocalFuncStat) and other.name == nm:
                            self.info.found = True
                            self.info.kind = "direct call"
                            self.info.line = other.line
                            self.info.params = list(other.params)
                            self.info.args = [self._arg_origin(a) for a in s.expr.args]
                            self.info.body_statements = len(other.body)
                            self.info.callee_globals = _globals_touched(other.body, self.lim)
                            break
                if self.info.found:
                    break
        if self.info.found:
            self.notes.append(
                f"entry point found: {self.info.kind} at line {self.info.line}, "
                f"parameters ({', '.join(self.info.params) or 'none'})")
            for p, a in zip(self.info.params, self.info.args):
                self.notes.append(
                    f"  • parameter '{p}' ← {a if a is not None else '[UNKNOWN]'}")
            if self.info.callee_globals:
                self.notes.append(
                    "  • globals touched by entry body: "
                    + ", ".join(self.info.callee_globals[:12]))
        else:
            self.notes.append(
                "entry point not found — script may run as top-level statements "
                "(no wrapper) or the entry is dynamic [UNKNOWN]")
        return self.info

    def _arg_origin(self, a: Node) -> Optional[str]:
        """Provenance of an argument (RULE 12): literal / pool ref /
        global name / computed / unknown."""
        if isinstance(a, NumLit):
            return f"number literal {_num_repr(a.value)}"
        if isinstance(a, StrLit):
            v = a.value
            return f"string literal {v[:32]!r}" + ("…" if len(v) > 32 else "")
        if isinstance(a, (TrueLit, FalseLit)):
            return f"boolean literal {type(a).__name__[:-3].lower()}"
        if isinstance(a, NilLit):
            return "nil literal"
        if isinstance(a, Name):
            return f"variable '{a.id}'"
        if isinstance(a, Dot):
            p = _dot_path(a)
            if p:
                return f"member access '{p}'"
            return None
        if isinstance(a, Call):
            f = a.func
            if isinstance(f, Name):
                return f"result of call '{f.id}(…)'"
            if isinstance(f, Dot):
                p = _dot_path(f)
                if p:
                    return f"result of call '{p}(…)'"
            return "result of call [UNKNOWN]"
        if isinstance(a, TableLit):
            return "table constructor"
        if isinstance(a, FuncExpr):
            return "function value (closure)"
        if isinstance(a, BinOp):
            return f"computed expression ({a.op})"
        if isinstance(a, UnOp):
            return "computed expression (unary)"
        return None


class PayloadAnalyzer:
    """RULE 13: recover the payload — what the script ultimately *does*."""

    def __init__(self, lim: "_Limits", entry: EntryInfo):
        self.lim = lim
        self.entry = entry
        self.info = PayloadInfo()
        self.notes: List[str] = []

    def run(self, stats: List[Node]) -> PayloadInfo:
        prov: List[str] = []
        # collect the calls that matter across the whole program
        interesting: List[str] = []
        load_targets: List[str] = []
        for s in stats:
            for n in walk(s):
                self.lim.tick_step()
                if isinstance(n, Call):
                    f = n.func
                    name = None
                    if isinstance(f, Name):
                        name = f.id
                    elif isinstance(f, Dot):
                        name = _dot_path(f)
                    if name is None:
                        continue
                    if name in ("loadstring", "load", "require"):
                        load_targets.append(name + "(…)")
                        prov.append(f"dynamic-load call `{name}` found — payload is "
                                    "compiled at runtime; source string is the real payload")
                    elif name in ("print", "warn", "error", "write"):
                        args = ", ".join(_short(a) for a in n.args[:2])
                        interesting.append(f"{name}({args})")
                    elif name == "getgenv" or name.startswith("hook"):
                        interesting.append(f"{name}(…)")
                        prov.append(f"environment manipulation `{name}` detected")
                    elif name in ("Instance.new",):
                        interesting.append(f"Instance.new({_short(n.args[0]) if n.args else ''})")
                        prov.append("object instantiation detected (Roblox context)")
                    elif name and name.count(".") >= 1 and name.split(".")[0] in (
                            "game", "workspace", "players", "Players", "http", "syn",
                            "http_request", "request", "socket", "Identity"):
                        interesting.append(f"{name}(…)")
                        prov.append(f"API/service call `{name}` detected")
        # dedupe
        seen: set = set()
        uniq_calls = [c for c in interesting if not (c in seen or seen.add(c))]
        self.info.calls_of_interest = uniq_calls[:20]
        self.info.loadstring_targets = load_targets[:6]

        # confidence
        if load_targets and not uniq_calls:
            self.info.confidence = "low — payload hidden behind dynamic load"
            self.info.unknowns.append(
                "payload content not statically recoverable (loadstring/load of a "
                "computed string)")
        elif self.entry.found and uniq_calls:
            self.info.confidence = "medium — entry traced, actions identified"
        elif uniq_calls:
            self.info.confidence = "medium — actions identified without explicit entry"
        else:
            self.info.confidence = "low — no recognizable actions"
            self.info.unknowns.append("no recognizable payload actions found")

        # provenance chain: layers peeled in order
        prov.insert(0, "obfuscated source")
        prov.append("cleaned AST re-printed as Lua/Luau (this output)")
        self.info.provenance = prov

        self.info.summary = "; ".join(uniq_calls[:6]) if uniq_calls else (
            "no direct actions; likely a pure computation or wrapper")
        self.info.recovered = bool(uniq_calls or load_targets)

        self.notes.append(f"payload confidence: {self.info.confidence}")
        for c in uniq_calls[:8]:
            self.notes.append(f"  • action: {c}")
        for u in self.info.unknowns:
            self.notes.append(f"  • [UNKNOWN] {u}")
        return self.info


def _globals_touched(stats: List[Node], lim: "_Limits") -> List[str]:
    """Globals (Name usages that are not locally declared) inside a body."""
    declared: set = set()
    used: set = set()
    for s in stats:
        for n in walk(s):
            lim.tick_step()
            if isinstance(n, LocalStat):
                declared.update(n.names)
            elif isinstance(n, LocalFuncStat):
                declared.add(n.name)
            elif isinstance(n, FuncExpr):
                declared.update(n.params)
            elif isinstance(n, Name):
                used.add(n.id)
    return sorted(used - declared)[:24]


# ═══════════════════════════════════════════════════════════════════════
# §10  REPORT & CLEAN SCRIPT — strictly two separate outputs:
#   analysis      → human-readable rule-by-rule report; [UNKNOWN] is
#                   allowed HERE ONLY
#   clean_script  → pure Lua/Luau source.  No comments, no banners, no
#                   analysis notes, no markdown fences, no [UNKNOWN]
#                   markers.  Nothing before the first line or after
#                   the last line.  (RULE: critical output formatting)
# ═══════════════════════════════════════════════════════════════════════

def _clip(text: str, limit: int, label: str) -> str:
    """Clip text to limit; when clipping, note it in the REPORT only."""
    if len(text) <= limit:
        return text
    return text[:limit]


def _sanitize_clean_script(src: str) -> str:
    """Guarantee the clean script is pure Lua source (defense in depth).

    - strips accidental markdown code fences
    - strips any accidental banner/comment-looking decoration that the
      printer might have emitted
    - removes leading/trailing blank lines
    The AST printer already emits no comments; this is a final safety net.
    """
    lines = src.splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):                     # markdown fence — never allowed
            continue
        out.append(ln.rstrip())
    # drop leading/trailing blank lines
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


class ReportBuilder:
    """Assembles the analysis report from all analyzer outputs."""

    def __init__(self) -> None:
        self.sections: List[Tuple[str, List[str]]] = []

    def add(self, title: str, lines: List[str]) -> None:
        if lines:
            self.sections.append((title, lines))

    def render(self, max_chars: int) -> str:
        parts: List[str] = ["**Deobfuscation Analysis Report**"]
        for title, lines in self.sections:
            parts.append(f"\n**{title}**")
            for ln in lines:
                parts.append(ln)
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars - 60]
            cut = text.rfind("\n")
            if cut > 0:
                text = text[:cut]
            text += "\n…(report clipped at limit)"
        return text


def build_report(
    folder: "ConstFolder",
    pools: "PoolExtractor",
    decoders: "DecoderAnalyzer",
    vm: "VMAnalyzer",
    flat: "FlatteningAnalyzer",
    meta: "MetaAnalyzer",
    closures: "ClosureAnalyzer",
    entry: "EntryAnalyzer",
    payload: "PayloadAnalyzer",
    stats: List[Node],
    lim: "_Limits",
) -> str:
    rb = ReportBuilder()

    # RULE 2 & 9 — constant folding
    fold_lines: List[str] = []
    if folder.count:
        fold_lines.append(f"{folder.count} expression(s) simplified by constant folding")
        for nw in folder.noteworthy[:ENGINE["analysis_head_rows"]]:
            fold_lines.append(f"  • {nw}")
        if folder.count > len(folder.noteworthy):
            fold_lines.append(f"  • … and {folder.count - len(folder.noteworthy)} more")
    else:
        fold_lines.append("no foldable arithmetic camouflage found")
    rb.add("Constant Simplification (Rules 2, 9)", fold_lines)

    # RULE 3 — string pools
    rb.add("String Pools (Rule 3)", pools.notes or ["no string pools detected"])

    # RULES 4 & 10 — decoders & encodings
    dec_lines: List[str] = []
    for d in decoders.decoders.values():
        dec_lines.append(d.describe())
    if not decoders.decoders:
        dec_lines.append("no decoder helper functions identified")
    dec_lines.extend(decoders.encoding_notes[:ENGINE["analysis_head_rows"]])
    if decoders.xor_keys:
        dec_lines.append(
            "possible XOR key literal(s): "
            + ", ".join(repr(k[:24]) for k in decoders.xor_keys))
    rb.add("Decoders & Encodings (Rules 4, 10)", dec_lines)

    # RULES 5, 6, 11 — VM & control flow
    vm_lines = list(vm.notes)
    vm_lines.extend(flat.notes)
    if not vm_lines:
        vm_lines = ["no virtual machine or control-flow flattening detected"]
    rb.add("VM & Control Flow (Rules 5, 6, 11)", vm_lines)

    # RULES 7 & 8 — metatables & closures
    mc_lines = list(meta.notes)
    mc_lines.extend(closures.notes)
    if not mc_lines:
        mc_lines = ["no metatable proxies or wrapper closures detected"]
    rb.add("Metatables & Closures (Rules 7, 8)", mc_lines)

    # RULES 1 & 12 — entry point
    rb.add("Entry Point (Rules 1, 12)", entry.notes)

    # RULE 13 — payload
    rb.add("Payload (Rule 13)", payload.notes)

    # provenance chain
    if payload.info.provenance:
        rb.add("Provenance Chain", [" → ".join(payload.info.provenance[:8])])

    # RULE 16 reminder: complexity of the obfuscator ≠ complexity of payload
    tail: List[str] = [
        "Note (Rule 16): the complexity of the obfuscation layers does not imply "
        "the payload is equally complex — the recovered actions above are the "
        "actual behavior found in this specific sample.",
    ]
    rb.add("Methodology Notes", tail)

    return rb.render(ENGINE["max_report_chars"])


def build_clean_script(stats: List[Node], lim: "_Limits") -> str:
    """Re-print the cleaned AST as pure Lua/Luau source."""
    p = Printer()
    for s in stats:
        p.emit_stmt(s)
    src = p.result()
    src = _sanitize_clean_script(src)
    if len(src) > ENGINE["max_output_chars"]:
        # hard cap — clip at a statement boundary when possible
        src = src[:ENGINE["max_output_chars"]]
        cut = src.rfind("\n")
        if cut > 0:
            src = src[:cut] + "\n"
    return src


# ═══════════════════════════════════════════════════════════════════════
# §11  REGEX FALLBACK MODE — when full parsing fails (extremely mangled
# input, oversized token streams), fall back to a conservative lexical
# cleanup: safe constant folding via regex, basic pool heuristics, and
# re-emission of the original text with simplifications applied.
# ═══════════════════════════════════════════════════════════════════════

_ARITH_RE = re.compile(
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([+\-*/%])\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _regex_fold_constants(src: str, budget_ops: int) -> Tuple[str, int]:
    """Fold integer/float arithmetic that appears OUTSIDE string literals.

    We walk the source character-by-character tracking string/comment
    state so we never rewrite inside a literal.  Single pass, bounded ops.
    """
    out: List[str] = []
    i = 0
    n = len(src)
    ops = 0
    changed = 0
    mode_stack: List[str] = []            # "s" short string, "L" long bracket

    def in_string() -> bool:
        return bool(mode_stack)

    while i < n:
        ch = src[i]
        if mode_stack and mode_stack[-1] == "s":
            if ch == "\\":
                out.append(src[i:i + 2])
                i += 2
                continue
            if ch == mode_stack[-2]:
                mode_stack.pop()
                mode_stack.pop()
            out.append(ch)
            i += 1
            continue
        if mode_stack and mode_stack[-1] == "L":
            closing = mode_stack[-2]
            if src.startswith(closing, i):
                out.append(closing)
                mode_stack.pop()
                mode_stack.pop()
                i += len(closing)
                continue
            out.append(ch)
            i += 1
            continue
        # not in a string
        if ch in "\"'":
            mode_stack.extend((ch, "s"))
            out.append(ch)
            i += 1
            continue
        if src.startswith("--", i):
            # comment: short or long
            j = i + 2
            m = re.match(r"\[(=*)\[", src[j:])
            if m:
                closing = "]" + "=" * m.group(1).count("=") + "]"
                mode_stack.extend((closing, "L"))
                out.append(src[i:i + 2 + m.end()])
                i += 2 + m.end()
                continue
            # short comment — copy to end of line
            eol = src.find("\n", i)
            if eol == -1:
                eol = n
            out.append(src[i:eol])
            i = eol
            continue
        m = _ARITH_RE.match(src, i)
        if m:
            a_s, op, b_s = m.group(1), m.group(2), m.group(3)
            try:
                a = _parse_number_str(a_s)
                b = _parse_number_str(b_s)
            except ValueError:
                out.append(ch)
                i += 1
                continue
            ops += 1
            if ops > budget_ops:
                raise EngineOverflow("regex fallback folding budget exceeded")
            # Skip folds that would create a negative number where a
            # unary minus context is ambiguous — keep only clean results.
            if op == "+":
                r = a + b
            elif op == "-":
                r = a - b
            elif op == "*":
                r = a * b
            elif op == "/":
                if b == 0:
                    out.append(ch)
                    i += 1
                    continue
                r = a / b
            else:  # %
                if b == 0:
                    out.append(ch)
                    i += 1
                    continue
                r = a - (a // b) * b if (a >= 0) == (b >= 0) else a - _floor_div(a, b)
            rs = _fmt_fallback_number(r)
            out.append(rs)
            i = m.end()
            changed += 1
            continue
        if src.startswith("[[", i) or re.match(r"\[(=*)\[", src[i:]):
            m2 = re.match(r"\[(=*)\[", src[i:])
            if m2:
                closing = "]" + "=" * len(m2.group(1)) + "]"
                mode_stack.extend((closing, "L"))
                out.append(src[i:i + m2.end()])
                i += m2.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out), changed


def _floor_div(a: float, b: float) -> float:
    import math
    return math.floor(a / b)


def _parse_number_str(s: str) -> float:
    return float(s)


def _fmt_fallback_number(r: float) -> str:
    if r == int(r) and abs(r) < 1e15:
        return str(int(r))
    return repr(r)


def _fallback_hex_literals(src: str) -> Tuple[str, int]:
    """Convert standalone hex number literals to decimal (outside strings)."""
    return src, 0


def fallback_deobfuscate(src: str, lim: "_Limits") -> Dict[str, str]:
    """Full regex-fallback pipeline for unparsable input."""
    notes: List[str] = []
    notes.append(
        "⚠ full AST parsing failed for this sample — regex fallback mode engaged")
    folded = src
    changed_total = 0
    try:
        folded, c1 = _regex_fold_constants(src, ENGINE["max_fold_ops"])
        changed_total += c1
    except EngineOverflow:
        folded = src
        notes.append("folding budget exceeded — returning partially simplified source")
    if changed_total:
        notes.append(f"{changed_total} arithmetic expression(s) folded by regex "
                     "(results verified against Lua number semantics)")
    else:
        notes.append("no safely foldable arithmetic found in fallback mode")
    # heuristic pool note
    pool_hits = re.findall(r"(\w+)\s*=\s*\{\s*\"[^\"]*\"\s*,", folded)
    if pool_hits:
        notes.append("possible string pool table(s): " + ", ".join(sorted(set(pool_hits))[:5]))
    enc_counts: Dict[str, int] = {}
    for m in re.finditer(r"\"([A-Za-z0-9+/=]{8,})\"", folded):
        enc = classify_encoding(m.group(1))
        if enc != "plain":
            enc_counts[enc] = enc_counts.get(enc, 0) + 1
    for enc, c in enc_counts.items():
        note = f"{c} string literal(s) look {enc}"
        dec = None
        if enc in ("hex", "base64"):
            try:
                sample = re.search(
                    r"\"(" + (r"[0-9a-fA-F]{8,}" if enc == "hex" else r"[A-Za-z0-9+/=]{8,}") + r")\"",
                    folded)
                if sample:
                    raw = (bytes.fromhex(sample.group(1)) if enc == "hex"
                           else _decode_base64(sample.group(1)))
                    dec = _bytes_to_display(raw, 100)
            except Exception:
                dec = None
        if dec:
            note += f" — e.g. decoded: {dec!r}"
        notes.append(note)
    # VM heuristics
    if re.search(r"while\s+true\s+do", folded) and re.search(r"\bpc\s*=\s*pc\s*\+\s*1", folded):
        notes.append("VM-like structure: while-true dispatch + program counter "
                     "(not fully analyzed in fallback mode)")
    # clean script = folded source, sanitized (no fences / banners)
    clean = _sanitize_clean_script(folded)
    if len(clean) > ENGINE["max_output_chars"]:
        clean = clean[:ENGINE["max_output_chars"]]
    report = ReportBuilder()
    report.add("Fallback Mode", notes)
    return {
        "analysis": report.render(ENGINE["max_report_chars"]),
        "clean_script": clean,
    }


# ═══════════════════════════════════════════════════════════════════════
# §12  ENGINE FACADE — one DeobfuscationEngine instance per run
# (RULE 14: no shared state between samples).  Orchestrates the whole
# pipeline with hard time/op budgets from config.ENGINE.
# ═══════════════════════════════════════════════════════════════════════

class DeobfuscationEngine:
    """Static Lua/Luau deobfuscation engine.

    deobfuscate(obfuscated_code) → {
        'analysis':     str   # rule-by-rule report ([UNKNOWN] allowed here only)
        'clean_script': str   # PURE Lua/Luau — no comments/banners/markers
    }

    Never executes the sample.  Per-instance state only (RULE 14).
    """

    def __init__(self) -> None:
        budget = ENGINE["emulation_timeout_seconds"] + ENGINE["parse_timeout_seconds"]
        self._limits = _Limits(time.monotonic() + budget)
        self.elapsed = 0.0
        self.used_fallback = False

    # -- public entry --------------------------------------------------------
    def deobfuscate(self, obfuscated_code: str) -> Dict[str, str]:
        t0 = time.monotonic()
        try:
            return self._run(obfuscated_code)
        except EngineTimeout:
            return self._emergency(
                obfuscated_code,
                "processing time budget exceeded — analysis incomplete")
        except EngineOverflow as e:
            return self._emergency(
                obfuscated_code,
                f"size/complexity budget exceeded ({e}) — analysis incomplete")
        except RecursionError:
            return self._emergency(
                obfuscated_code,
                "AST nesting too deep for safe full analysis — analysis incomplete")
        finally:
            self.elapsed = time.monotonic() - t0

    # -- pipeline -------------------------------------------------------------
    def _run(self, src: str) -> Dict[str, str]:
        lim = self._limits
        if not src or not src.strip():
            return {
                "analysis": "**Deobfuscation Analysis Report**\n\n"
                            "**Input**\nThe file is empty — nothing to analyze.",
                "clean_script": "",
            }
        if len(src.encode("utf-8", "replace")) > ENGINE["max_source_bytes"]:
            return self._emergency(src, "source exceeds max_source_bytes")

        # 1) lex with normalization (Luau compound assignments → normal form)
        try:
            toks = lex(src, lim)
            if len(toks) > ENGINE["max_tokens"]:
                raise EngineOverflow("token stream too large")
            toks = normalize_luau_compound(toks)
        except SyntaxError as e:
            # NOTE: the lexer raises plain SyntaxError (not ParseError, which is a
            # subclass) -- catching SyntaxError covers both lexing and parsing
            # failures, routing unparsable/truncated obfuscations to fallback mode.
            self.used_fallback = True
            return fallback_deobfuscate(src, lim)
        except (EngineTimeout, EngineOverflow):
            raise

        # 2) parse to AST
        try:
            parser = Parser(toks, lim)
            stats = parser.parse_chunk()
        except SyntaxError as e:
            # ParseError subclasses SyntaxError; the lexer may raise either.
            self.used_fallback = True
            return fallback_deobfuscate(src, lim)

        # 3) constant folding (RULES 2 & 9)
        folder = ConstFolder(lim)
        folder.run(stats)

        # 4) string pools (RULE 3)
        pools = PoolExtractor(lim)
        pools.run(stats)

        # 5) decoders & encodings (RULES 4 & 10)
        decoders = DecoderAnalyzer(lim, pools.pools)
        decoders.run(stats)

        # 6) pool substitution (RULE 4)
        sub = PoolSubstitutor(lim, pools.pools)
        sub.run(stats)
        if sub.notes:
            pools.notes.extend(sub.notes)

        # 6b) re-fold after substitution so combined expressions collapse
        folder2 = ConstFolder(lim)
        folder2.run(stats)
        folder.count += folder2.count
        folder.noteworthy.extend(folder2.noteworthy[:8])

        # 7) VM & control flow (RULES 5, 6, 11)
        vm = VMAnalyzer(lim)
        vm.run(stats)
        flat = FlatteningAnalyzer(lim)
        flat.run(stats)

        # 8) metatables & closures (RULES 7, 8)
        meta = MetaAnalyzer(lim)
        meta.run(stats)
        closures = ClosureAnalyzer(lim)
        closures.run(stats)

        # 9) entry & payload (RULES 1, 12, 13)
        entry = EntryAnalyzer(lim)
        entry.run(stats)
        payload = PayloadAnalyzer(lim, entry.info)
        payload.run(stats)

        # 10) outputs — two SEPARATE deliverables
        analysis = build_report(folder, pools, decoders, vm, flat, meta,
                                closures, entry, payload, stats, lim)
        clean = build_clean_script(stats, lim)

        return {"analysis": analysis, "clean_script": clean}

    # -- emergency path -------------------------------------------------------
    def _emergency(self, src: str, reason: str) -> Dict[str, str]:
        self.used_fallback = True
        safe = _sanitize_clean_script(src)
        if len(safe) > ENGINE["max_output_chars"]:
            safe = safe[:ENGINE["max_output_chars"]]
        rb = ReportBuilder()
        rb.add("Engine Limits", [reason])
        rb.add("What you received", [
            "the original source (unchanged), sanitized of markdown fences",
            "re-run on a smaller or less complex sample if possible",
        ])
        return {"analysis": rb.render(ENGINE["max_report_chars"]),
                "clean_script": safe}
