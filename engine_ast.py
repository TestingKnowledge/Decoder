"""
engine_ast.py — Lua/Luau AST nodes, recursive-descent parser and pretty-printer.

Consumed by deobfuscator.py. The parser is hand-written (no external deps)
and covers Lua 5.1–5.4 syntax plus the Luau dialect (compound assignment,
'continue', type annotations are *skipped* transparently).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lua_base import Tok, lex, _Limits, EngineOverflow, ENGINE, LuaValue, LuaTable, _quote_lua, _fmt_number


# ══════════════════════════════════════════════════════════════════════════════
# AST NODES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    line: int = 0
    kind: str = ""


# ── expressions ──────────────────────────────────────────────────────────────

@dataclass
class NilLit(Node):
    pass


@dataclass
class TrueLit(Node):
    pass


@dataclass
class FalseLit(Node):
    pass


@dataclass
class Vararg(Node):
    pass


@dataclass
class NumLit(Node):
    value: Any = None            # int | float


@dataclass
class StrLit(Node):
    value: str = ""


@dataclass
class Name(Node):
    id: str = ""


@dataclass
class Index(Node):               # obj[key]  (bracket)
    obj: Node = None
    key: Node = None


@dataclass
class Dot(Node):                 # obj.name
    obj: Node = None
    name: str = ""


@dataclass
class Call(Node):                # f(args...)
    func: Node = None
    args: List[Node] = field(default_factory=list)


@dataclass
class Invoke(Node):              # obj:method(args...)
    obj: Node = None
    method: str = ""
    args: List[Node] = field(default_factory=list)


@dataclass
class FuncExpr(Node):            # function(params) body end
    params: List[str] = field(default_factory=list)
    is_vararg: bool = False
    body: List[Node] = field(default_factory=list)


@dataclass
class TableLit(Node):
    entries: List[Tuple[Optional[Node], Node]] = field(default_factory=list)
    # entry key None → positional; key is [k, v] pair node


@dataclass
class BinOp(Node):
    op: str = ""
    left: Node = None
    right: Node = None


@dataclass
class UnOp(Node):
    op: str = ""
    operand: Node = None


# ── statements ───────────────────────────────────────────────────────────────

@dataclass
class LocalStat(Node):
    names: List[str] = field(default_factory=list)
    exprs: List[Node] = field(default_factory=list)


@dataclass
class AssignStat(Node):
    targets: List[Node] = field(default_factory=list)
    exprs: List[Node] = field(default_factory=list)


@dataclass
class CallStat(Node):            # expression-statement that is a call
    expr: Node = None


@dataclass
class DoStat(Node):
    body: List[Node] = field(default_factory=list)


@dataclass
class WhileStat(Node):
    cond: Node = None
    body: List[Node] = field(default_factory=list)


@dataclass
class RepeatStat(Node):
    body: List[Node] = field(default_factory=list)
    cond: Node = None


@dataclass
class IfStat(Node):
    cond: Node = None
    then_body: List[Node] = field(default_factory=list)
    elseifs: List[Tuple[Node, List[Node]]] = field(default_factory=list)
    else_body: Optional[List[Node]] = None


@dataclass
class NumForStat(Node):
    var: str = ""
    start: Node = None
    stop: Node = None
    step: Optional[Node] = None
    body: List[Node] = field(default_factory=list)


@dataclass
class GenForStat(Node):
    names: List[str] = field(default_factory=list)
    exprs: List[Node] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class FuncStat(Node):            # function a.b:c() ... end
    name: Node = None            # Name/Dot/Index chain
    is_local: bool = False
    is_method: bool = False      # declared with ':' (implicit self)
    params: List[str] = field(default_factory=list)
    is_vararg: bool = False
    body: List[Node] = field(default_factory=list)


@dataclass
class ReturnStat(Node):
    exprs: List[Node] = field(default_factory=list)


@dataclass
class BreakStat(Node):
    pass


@dataclass
class GotoStat(Node):
    label: str = ""


@dataclass
class LabelStat(Node):
    label: str = ""


@dataclass
class LocalFuncStat(Node):
    name: str = ""
    params: List[str] = field(default_factory=list)
    is_vararg: bool = False
    body: List[Node] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════════════════════

class ParseError(SyntaxError):
    pass


class Parser:
    # Nesting guard: the recursive-descent chain costs ~15 stack frames per
    # expression level, and the engine raises Python's limit to 30_000 (pure
    # Python frames in 3.11+ are heap-allocated).  1500 levels * ~15 frames
    # stays safely inside that budget; deeper input routes to fallback mode.
    MAX_EXPR_DEPTH = 1_500

    def __init__(self, toks: List[Tok], lim: _Limits):
        self.toks = toks
        self.i = 0
        self.lim = lim
        self._depth = 0

    # ── token helpers ───────────────────────────────────────────────────────
    def peek(self, k: int = 0) -> Tok:
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def next(self) -> Tok:
        t = self.toks[self.i]
        if t.type != "eof":
            self.i += 1
        return t

    def check(self, type_: str, value: Any = None) -> bool:
        t = self.peek()
        return t.type == type_ and (value is None or t.value == value)

    def accept(self, type_: str, value: Any = None) -> Optional[Tok]:
        if self.check(type_, value):
            return self.next()
        return None

    def expect(self, type_: str, value: Any = None) -> Tok:
        t = self.peek()
        if t.type == type_ and (value is None or t.value == value):
            return self.next()
        want = value if value is not None else type_
        got = t.raw if t.raw else str(t.value)
        raise ParseError(f"line {t.line}: expected {want!r}, got {got!r}")

    # ── entry ───────────────────────────────────────────────────────────────
    def parse_chunk(self) -> List[Node]:
        stats = self.parse_block(stop={"eof"})
        self.expect("eof")
        return stats

    def parse_block(self, stop: set) -> List[Node]:
        stats: List[Node] = []
        while True:
            self.lim.tick_node()
            t = self.peek()
            if t.type == "eof":
                break
            if t.type == "keyword" and t.value in stop:
                break
            if t.type == "op" and t.value in stop:
                break
            if t.type == "keyword" and t.value == "return":
                stats.append(self.parse_return())
                break  # return terminates a block
            stats.append(self.parse_statement())
        return stats

    # ── statements ──────────────────────────────────────────────────────────
    def parse_statement(self) -> Node:
        t = self.peek()
        line = t.line
        if t.type == "keyword":
            kw = t.value
            if kw == ";":
                self.next()
                return DoStat(line=line, body=[])
            if kw == "local":
                return self.parse_local()
            if kw == "if":
                return self.parse_if()
            if kw == "while":
                return self.parse_while()
            if kw == "do":
                self.next()
                body = self.parse_block(stop={"end"})
                self.expect("keyword", "end")
                return DoStat(line=line, body=body)
            if kw == "for":
                return self.parse_for()
            if kw == "function":
                return self.parse_funcstat()
            if kw == "repeat":
                self.next()
                body = self.parse_block(stop={"until"})
                self.expect("keyword", "until")
                cond = self.parse_expr()
                return RepeatStat(line=line, body=body, cond=cond)
            if kw == "break":
                self.next()
                return BreakStat(line=line)
            if kw == "goto":
                self.next()
                lab = self.expect("name").value
                return GotoStat(line=line, label=lab)
            if kw == "continue":  # Luau
                self.next()
                # Represent as a Goto to ::__continue__ for analysis purposes
                return GotoStat(line=line, label="__continue__")
        if t.type == "op":
            if t.value == "::":
                self.next()
                lab = self.expect("name").value
                self.expect("op", "::")
                return LabelStat(line=line, label=lab)
            if t.value == ";":
                self.next()
                return DoStat(line=line, body=[])
        # expression statement (call) or assignment
        return self.parse_exprstat()

    def parse_local(self) -> Node:
        self.expect("keyword", "local")
        line = self.peek().line
        if self.check("keyword", "function"):
            self.next()
            name = self.expect("name").value
            params, vararg, body = self.parse_funcbody()
            return LocalFuncStat(line=line, name=name, params=params,
                                 is_vararg=vararg, body=body)
        names = [self._expect_name_with_attr()]
        while self.accept("op", ","):
            names.append(self._expect_name_with_attr())
        exprs: List[Node] = []
        if self.accept("op", "="):
            exprs = self.parse_exprlist()
        return LocalStat(line=line, names=names, exprs=exprs)

    def _expect_name_with_attr(self) -> str:
        name = self.expect("name").value
        # Lua 5.4 attribute: <const> / <close> (parsed & dropped)
        if (self.check("op", "<") and self.peek(1).type == "name"
                and self.peek(2).type == "op" and self.peek(2).value == ">"):
            self.next(); self.next(); self.next()
        return name

    def parse_funcstat(self) -> Node:
        self.expect("keyword", "function")
        line = self.peek().line
        # funcname: Name {'.' Name} [':' Name]
        name: Node = Name(line=line, id=self.expect("name").value)
        is_method = False
        while self.check("op", "."):
            self.next()
            name = Dot(line=line, obj=name, name=self.expect("name").value)
        if self.check("op", ":"):
            self.next()
            is_method = True
            name = Dot(line=line, obj=name, name=self.expect("name").value)
        params, vararg, body = self.parse_funcbody()
        return FuncStat(line=line, name=name, params=params,
                        is_vararg=vararg, is_method=is_method, body=body)

    def parse_funcbody(self) -> Tuple[List[str], bool, List[Node]]:
        self._skip_generics()
        self.expect("op", "(")
        params: List[str] = []
        vararg = False
        if not self.check("op", ")"):
            while True:
                if self.check("op", "..."):
                    self.next()
                    vararg = True
                    # Luau: `...: T`
                    if self.check("op", ":"):
                        self._skip_type()
                    break
                params.append(self.expect("name").value)
                if self.check("op", ":"):     # Luau param type annotation
                    self._skip_type()
                if self.accept("op", ","):
                    continue
                break
        self.expect("op", ")")
        if self.check("op", ":"):             # Luau return type annotation
            self._skip_type()
        body = self.parse_block(stop={"end"})
        self.expect("keyword", "end")
        return params, vararg, body

    def _skip_generics(self) -> None:
        """Skip Luau generic type parameter list `<T, U...>` before '('."""
        if self.check("op", "<"):
            depth = 0
            while True:
                t = self.peek()
                if t.type == "eof":
                    return
                if t.type == "op" and t.value == "<":
                    depth += 1
                    self.next()
                    continue
                if t.type == "op" and t.value == ">":
                    depth -= 1
                    self.next()
                    if depth <= 0:
                        return
                    continue
                if t.type == "op" and t.value in ("(", "[", "{"):
                    depth += 1
                    self.next()
                    continue
                if t.type == "op" and t.value in (")", "]", "}"):
                    depth -= 1
                    self.next()
                    continue
                if depth <= 0 and t.type == "op" and t.value not in (",", ".", "::", "|", "?", "..."):
                    return
                self.next()

    def _skip_type(self) -> None:
        """Consume a Luau type expression (best-effort, balanced)."""
        depth = 0
        while True:
            t = self.peek()
            if t.type == "eof":
                return
            if t.type == "op" and t.value in ("(", "[", "{", "<"):
                depth += 1
                self.next()
                continue
            if t.type == "op" and t.value in (")", "]", "}", ">"):
                if depth == 0:
                    return
                depth -= 1
                self.next()
                continue
            if t.type == "op" and t.value in (",",):
                if depth == 0:
                    return
                self.next()
                continue
            if t.type == "keyword" and t.value in ("end", "local", "return",
                                                   "if", "while", "for", "do",
                                                   "function", "repeat", "until", "break", "goto"):
                return
            if t.type == "op" and t.value == "->":
                self.next()
                continue
            if t.type == "name" or t.type == "keyword" and t.value == "nil":
                self.next()
                continue
            if t.type == "string":
                self.next()
                continue
            if t.type == "op" and t.value in ("?", "|", "&", "...", "::", ".", ":"):
                self.next()
                continue
            if t.type == "op" and t.value == "->":
                self.next()
                continue
            return

    def parse_if(self) -> Node:
        line = self.peek().line
        self.expect("keyword", "if")
        cond = self.parse_expr()
        self.expect("keyword", "then")
        then_body = self.parse_block(stop={"elseif", "else", "end"})
        elseifs: List[Tuple[Node, List[Node]]] = []
        else_body: Optional[List[Node]] = None
        while self.check("keyword", "elseif"):
            self.next()
            c = self.parse_expr()
            self.expect("keyword", "then")
            b = self.parse_block(stop={"elseif", "else", "end"})
            elseifs.append((c, b))
        if self.check("keyword", "else"):
            self.next()
            else_body = self.parse_block(stop={"end"})
        self.expect("keyword", "end")
        return IfStat(line=line, cond=cond, then_body=then_body,
                      elseifs=elseifs, else_body=else_body)

    def parse_while(self) -> Node:
        line = self.peek().line
        self.expect("keyword", "while")
        cond = self.parse_expr()
        self.expect("keyword", "do")
        body = self.parse_block(stop={"end"})
        self.expect("keyword", "end")
        return WhileStat(line=line, cond=cond, body=body)

    def parse_for(self) -> Node:
        line = self.peek().line
        self.expect("keyword", "for")
        first = self.expect("name").value
        if self.check("op", "="):
            self.next()
            start = self.parse_expr()
            self.expect("op", ",")
            stop = self.parse_expr()
            step = None
            if self.accept("op", ","):
                step = self.parse_expr()
            self.expect("keyword", "do")
            body = self.parse_block(stop={"end"})
            self.expect("keyword", "end")
            return NumForStat(line=line, var=first, start=start, stop=stop,
                              step=step, body=body)
        names = [first]
        while self.accept("op", ","):
            names.append(self.expect("name").value)
        self.expect("keyword", "in")
        exprs = self.parse_exprlist()
        self.expect("keyword", "do")
        body = self.parse_block(stop={"end"})
        self.expect("keyword", "end")
        return GenForStat(line=line, names=names, exprs=exprs, body=body)

    def parse_return(self) -> Node:
        line = self.peek().line
        self.expect("keyword", "return")
        exprs: List[Node] = []
        if not (self.check("keyword", "end") or self.check("op", ";")
                or self.check("eof") or self._at_block_stop()):
            exprs = self.parse_exprlist()
        self.accept("op", ";")
        return ReturnStat(line=line, exprs=exprs)

    def _at_block_stop(self) -> bool:
        t = self.peek()
        if t.type == "keyword" and t.value in ("end", "else", "elseif", "until"):
            return True
        return False

    def parse_exprstat(self) -> Node:
        t = self.peek()
        line = t.line
        # Parse a suffixed expression; a call/invoke → CallStat,
        # otherwise it must be an assignment target.
        e = self._parse_suffixed()
        if isinstance(e, (Call, Invoke)):
            if self.check("op", ","):
                raise ParseError(f"line {line}: unexpected ',' after call")
            return CallStat(line=line, expr=e)
        # assignment
        targets = [e]
        while self.accept("op", ","):
            targets.append(self.parse_suffixed_target())
        self.expect("op", "=")
        exprs = self.parse_exprlist()
        return AssignStat(line=line, targets=targets, exprs=exprs)

    def parse_suffixed_target(self) -> Node:
        e = self._parse_suffixed()
        if isinstance(e, (Call, Invoke)):
            raise ParseError(f"line {e.line}: cannot assign to a call")
        return e

    # ── expressions ──────────────────────────────────────────────────────────
    def parse_exprlist(self) -> List[Node]:
        exprs = [self.parse_expr()]
        while self.accept("op", ","):
            exprs.append(self.parse_expr())
        return exprs

    def parse_expr(self) -> Node:
        self._depth += 1
        try:
            if self._depth > self.MAX_EXPR_DEPTH:
                raise ParseError(
                    f"expression nesting deeper than {self.MAX_EXPR_DEPTH} "
                    "levels (parser guard)")
            return self._parse_or()
        finally:
            self._depth -= 1

    def _parse_or(self) -> Node:
        left = self._parse_and()
        while self.check("keyword", "or"):
            line = self.peek().line
            self.next()
            right = self._parse_and()
            left = BinOp(line=line, op="or", left=left, right=right)
        return left

    def _parse_and(self) -> Node:
        left = self._parse_cmp()
        while self.check("keyword", "and"):
            line = self.peek().line
            self.next()
            right = self._parse_cmp()
            left = BinOp(line=line, op="and", left=left, right=right)
        return left

    def _parse_cmp(self) -> Node:
        left = self._parse_bor()
        while self.peek().type == "op" and self.peek().value in ("<", ">", "<=", ">=", "==", "~="):
            line = self.peek().line
            op = self.next().value
            right = self._parse_bor()
            left = BinOp(line=line, op=op, left=left, right=right)
        return left

    def _parse_bor(self) -> Node:
        left = self._parse_bxor()
        while self.peek().type == "op" and self.peek().value == "|":
            line = self.peek().line
            self.next()
            left = BinOp(line=line, op="|", left=left, right=self._parse_bxor())
        return left

    def _parse_bxor(self) -> Node:
        left = self._parse_band()
        while self.peek().type == "op" and self.peek().value == "~":
            line = self.peek().line
            self.next()
            left = BinOp(line=line, op="~", left=left, right=self._parse_band())
        return left

    def _parse_band(self) -> Node:
        left = self._parse_shift()
        while self.peek().type == "op" and self.peek().value == "&":
            line = self.peek().line
            self.next()
            left = BinOp(line=line, op="&", left=left, right=self._parse_shift())
        return left

    def _parse_shift(self) -> Node:
        left = self._parse_concat()
        while self.peek().type == "op" and self.peek().value in ("<<", ">>"):
            line = self.peek().line
            op = self.next().value
            left = BinOp(line=line, op=op, left=left, right=self._parse_concat())
        return left

    def _parse_concat(self) -> Node:
        left = self._parse_add()
        while self.peek().type == "op" and self.peek().value == "..":
            line = self.peek().line
            self.next()
            left = BinOp(line=line, op="..", left=left, right=self._parse_add())
        return left

    def _parse_add(self) -> Node:
        left = self._parse_mul()
        while self.peek().type == "op" and self.peek().value in ("+", "-"):
            line = self.peek().line
            op = self.next().value
            left = BinOp(line=line, op=op, left=left, right=self._parse_mul())
        return left

    def _parse_mul(self) -> Node:
        left = self._parse_unary()
        while self.peek().type == "op" and self.peek().value in ("*", "/", "//", "%"):
            line = self.peek().line
            op = self.next().value
            left = BinOp(line=line, op=op, left=left, right=self._parse_unary())
        return left

    def _parse_unary(self) -> Node:
        t = self.peek()
        if (t.type == "op" and t.value in ("-", "#", "~")) or \
           (t.type == "keyword" and t.value == "not"):
            line = t.line
            self.next()
            operand = self._parse_unary()
            return UnOp(line=line, op=t.value, operand=operand)
        return self._parse_pow()

    def _parse_simple(self) -> Node:
        """Simple expressions: literals, vararg, table constructors."""
        t = self.peek()
        if t.type == "keyword" and t.value == "nil":
            self.next()
            return NilLit(line=t.line)
        if t.type == "keyword" and t.value == "true":
            self.next()
            return TrueLit(line=t.line)
        if t.type == "keyword" and t.value == "false":
            self.next()
            return FalseLit(line=t.line)
        if t.type == "number":
            self.next()
            return NumLit(line=t.line, value=t.value)
        if t.type == "string":
            self.next()
            return StrLit(line=t.line, value=t.value)
        if t.type == "op" and t.value == "...":
            self.next()
            return Vararg(line=t.line)
        if t.type == "op" and t.value == "{":
            return self.parse_table()
        return self._parse_suffixed()

    def _parse_pow(self) -> Node:
        base = self._parse_simple()
        if self.peek().type == "op" and self.peek().value == "^":
            line = self.peek().line
            self.next()
            exp = self._parse_unary()   # ^ is right-assoc & binds tighter than unary
            return BinOp(line=line, op="^", left=base, right=exp)
        return base

    def _parse_suffixed(self) -> Node:
        e = self._parse_primary()
        while True:
            t = self.peek()
            if t.type == "op" and t.value == ".":
                self.next()
                name = self.expect("name").value
                e = Dot(line=t.line, obj=e, name=name)
            elif t.type == "op" and t.value == "[":
                self.next()
                key = self.parse_expr()
                self.expect("op", "]")
                e = Index(line=t.line, obj=e, key=key)
            elif t.type == "op" and t.value == "(":
                self.next()
                args = self._parse_call_args_rest()
                e = Call(line=t.line, func=e, args=args)
            elif t.type == "string":
                s = self.next()
                e = Call(line=t.line, func=e, args=[StrLit(line=t.line, value=s.value)])
            elif t.type == "op" and t.value == "{":
                e = self._parse_table_call(e, t)
            elif t.type == "op" and t.value == ":":
                self.next()
                m = self.expect("name").value
                if self.check("op", "("):
                    self.next()
                    args = self._parse_call_args_rest()
                elif self.check("string") or self.check("op", "{"):
                    args = [self._parse_primary_arg()]
                else:
                    args = []
                e = Invoke(line=t.line, obj=e, method=m, args=args)
            else:
                break
        return e

    def _parse_primary(self) -> Node:
        t = self.peek()
        if t.type == "name":
            self.next()
            return Name(line=t.line, id=t.value)
        if t.type == "keyword" and t.value == "function":
            self.next()
            params, vararg, body = self.parse_funcbody()
            return FuncExpr(line=t.line, params=params, is_vararg=vararg, body=body)
        if t.type == "op" and t.value == "(":
            # parenthesized expression — canonicalized (printer re-adds
            # parens from precedence, so the node is returned unwrapped)
            self.next()
            e = self.parse_expr()
            self.expect("op", ")")
            return e
        raise ParseError(f"line {t.line}: unexpected symbol {t.raw or t.value!r} "
                         "in expression")

    def _parse_primary_arg(self) -> Node:
        """A single primary expression valid as a sole method-call argument."""
        t = self.peek()
        if t.type == "string":
            self.next()
            return StrLit(line=t.line, value=t.value)
        if t.type == "op" and t.value == "{":
            return self.parse_table()
        raise ParseError(f"line {t.line}: expected string or table argument")

    def _parse_call_args_rest(self) -> List[Node]:
        """Parse the remainder of a call after '(' has been consumed."""
        args: List[Node] = []
        if not self.check("op", ")"):
            args = self.parse_exprlist()
        self.expect("op", ")")
        return args

    def _parse_table_call(self, e: Node, t: Tok) -> Node:
        tbl = self.parse_table()
        return Call(line=t.line, func=e, args=[tbl])

    def parse_table(self) -> Node:
        t = self.expect("op", "{")
        line = t.line
        entries: List[Tuple[Optional[Node], Node]] = []
        while not self.check("op", "}"):
            self.lim.tick_node()
            if self.check("op", "["):
                self.next()
                key = self.parse_expr()
                self.expect("op", "]")
                self.expect("op", "=")
                val = self.parse_expr()
                entries.append((key, val))
            elif self.check("name") and self.peek(1).type == "op" and self.peek(1).value == "=":
                name = self.next().value
                self.next()  # '='
                val = self.parse_expr()
                entries.append((StrLit(line=line, value=name), val))
            else:
                val = self.parse_expr()
                entries.append((None, val))
            if not (self.accept("op", ",") or self.accept("op", ";")):
                break
        self.expect("op", "}")
        return TableLit(line=line, entries=entries)


# ═════════════════════════ IFs ══════════════════════════════════════════════
# compound assignment (Luau) is normalized to plain assignment
# ════════════════════════════════════════════════════════════════════════════

_LUAU_COMPOUND = {"+=", "-=", "*=", "/=", "%=", "^=", "..=", "//="}


def normalize_luau_compound(toks):
    """Rewrite Luau `target op= expr` tokens into `target = target op expr`.

    Statement-level only: fires when a compound op follows a suffixed
    target expression. The backward scan respects statement boundaries:
    at depth 0, ')' '}' '(' '{' '=' and keywords terminate the target,
    while ']' opens an index bracket that belongs to the target itself.
    Operates on a copy; the original token list is untouched.
    """
    out = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t.type == "op" and t.value == "===":
            out.append(Tok("op", "==", t.pos, t.line, t.col, "=="))
            i += 1
            continue
        if t.type == "op" and t.value == "!==":
            out.append(Tok("op", "~=", t.pos, t.line, t.col, "~="))
            i += 1
            continue
        if t.type == "op" and t.value in _LUAU_COMPOUND:
            # ── backward scan for the assignment target ──
            k = len(out) - 1
            depth = 0
            while k >= 0:
                u = out[k]
                if u.type == "keyword":
                    break
                v = u.value if u.type == "op" else None
                if v == "]":
                    depth += 1
                elif v == "[":
                    # '[' at depth 0 opens an index that belongs to the
                    # target itself → consume it, do not break
                    if depth == 0:
                        depth += 1
                    else:
                        depth -= 1
                elif v == ")":
                    if depth == 0:
                        break          # call end / prev statement
                    depth += 1
                elif v == "(":
                    if depth == 0:
                        break
                    depth -= 1
                elif v == "}" or v == "{":
                    # tables never appear inside an assignable target chain
                    break
                elif v == "=" and depth == 0:
                    break              # previous assignment
                elif v in (".", ":") and depth == 0:
                    pass               # part of a.b / a:b chain
                elif u.type == "name" and depth == 0:
                    pass               # part of target chain
                elif depth > 0:
                    pass               # inside brackets — anything goes
                else:
                    break              # number/string/','/etc → not a target
                k -= 1
            start = k + 1
            target_toks = out[start:]
            ok = bool(target_toks) and target_toks[0].type == "name" and depth == 0
            if ok and len(target_toks) > 1 and target_toks[-1].type == "op" \
                    and target_toks[-1].value in (")", "}"):
                # ')' → call, '}' → table: not assignable targets.
                # ']' is a legit bracketed-index target and is accepted.
                ok = False
            if not ok:
                out.append(t)
                i += 1
                continue
            op = t.value[:-1]          # '+=' → '+'
            out = out[:start]
            out.extend(target_toks)
            out.append(Tok("op", "=", t.pos, t.line, t.col, "="))
            out.extend(target_toks)
            out.append(Tok("op", op, t.pos, t.line, t.col, op))
            i += 1
            continue
        out.append(t)
        i += 1
    return out


# =====================================================================
# PRETTY-PRINTER (beautifier)
# ══════════════════════════════════════════════════════════════════════════════

_PREC = {
    "or": 1, "and": 2,
    "<": 3, ">": 3, "<=": 3, ">=": 3, "==": 3, "~=": 3,
    "|": 4, "~": 5, "&": 6, "<<": 7, ">>": 7,
    "..": 8, "+": 9, "-": 9, "*": 10, "/": 10, "//": 10, "%": 10,
    "unary": 11, "^": 12,
}


class Printer:
    """Serialize the AST back to clean, properly indented Lua/Luau."""

    MAX_CALL_INLINE = 3   # args above this → one arg per line

    def __init__(self):
        self.lines: List[str] = []
        self.indent = 0
        self.emitted_nodes = 0

    def emit(self, text: str = ""):
        self.lines.append("    " * self.indent + text if text else "")

    def emit_stmt(self, s: Node):
        self.emitted_nodes += 1
        if isinstance(s, LocalStat):
            if s.exprs:
                self.emit(f"local {_join_names(s.names)} = {self.expr_list(s.exprs)}")
            else:
                self.emit(f"local {_join_names(s.names)}")
        elif isinstance(s, AssignStat):
            self.emit(f"{self.expr_list(s.targets)} = {self.expr_list(s.exprs)}")
        elif isinstance(s, CallStat):
            self.emit(self.expr(s.expr))
        elif isinstance(s, DoStat):
            if s.body:
                self.emit("do")
                self._indented(s.body)
                self.emit("end")
            else:
                pass  # empty do → skip entirely
        elif isinstance(s, WhileStat):
            self.emit(f"while {self.expr(s.cond)} do")
            self._indented(s.body)
            self.emit("end")
        elif isinstance(s, RepeatStat):
            self.emit("repeat")
            self._indented(s.body)
            self.emit(f"until {self.expr(s.cond)}")
        elif isinstance(s, IfStat):
            self._emit_if(s)
        elif isinstance(s, NumForStat):
            step = f", {self.expr(s.step)}" if s.step is not None else ""
            self.emit(f"for {s.var} = {self.expr(s.start)}, {self.expr(s.stop)}{step} do")
            self._indented(s.body)
            self.emit("end")
        elif isinstance(s, GenForStat):
            self.emit(f"for {_join_names(s.names)} in {self.expr_list(s.exprs)} do")
            self._indented(s.body)
            self.emit("end")
        elif isinstance(s, FuncStat):
            name = self._funcname(s.name, s.is_method)
            self._emit_func(f"function {name}", s.params, s.body, s.is_vararg)
        elif isinstance(s, LocalFuncStat):
            self._emit_func(f"local function {s.name}", s.params, s.body, s.is_vararg)
        elif isinstance(s, ReturnStat):
            if s.exprs:
                self.emit(f"return {self.expr_list(s.exprs)}")
            else:
                self.emit("return")
        elif isinstance(s, BreakStat):
            self.emit("break")
        elif isinstance(s, GotoStat):
            if s.label == "__continue__":
                self.emit("continue")
            else:
                self.emit(f"goto {s.label}")
        elif isinstance(s, LabelStat):
            self.emit(f"::{s.label}::")
        else:
            # unknown statement → skip silently to keep output valid
            pass

    def _funcname(self, n: Node, is_method: bool) -> str:
        """Render a function-declaration name chain, preserving ':' methods."""
        if isinstance(n, Name):
            return n.id
        if isinstance(n, Dot):
            base = self._funcname(n.obj, False)
            sep = ":" if is_method else "."
            return f"{base}{sep}{n.name}"
        if isinstance(n, Index):
            return f"{self._suffixed_atom(n.obj)}[{self.expr(n.key)}]"
        return self.expr(n)

    def _emit_func(self, header: str, params: List[str], body: List[Node], vararg: bool):
        plist = ", ".join(params + (["..."] if vararg else []))
        self.emit(f"{header}({plist})")
        self._indented(body)
        self.emit("end")

    def _emit_if(self, s: IfStat):
        self.emit(f"if {self.expr(s.cond)} then")
        self._indented(s.then_body)
        for cond, blk in s.elseifs:
            self.emit(f"elseif {self.expr(cond)} then")
            self._indented(blk)
        if s.else_body is not None:
            self.emit("else")
            self._indented(s.else_body)
        self.emit("end")

    def _indented(self, body: List[Node]):
        self.indent += 1
        for s in body:
            self.emit_stmt(s)
        self.indent -= 1

    # ── expressions ─────────────────────────────────────────────────────────
    def expr_list(self, nodes: List[Node]) -> str:
        return ", ".join(self.expr(n) for n in nodes)

    def expr(self, n: Node, parent_prec: int = 0, side: str = "left") -> str:
        self.emitted_nodes += 1
        if isinstance(n, NilLit):
            return "nil"
        if isinstance(n, TrueLit):
            return "true"
        if isinstance(n, FalseLit):
            return "false"
        if isinstance(n, Vararg):
            return "..."
        if isinstance(n, NumLit):
            return _fmt_num(n.value)
        if isinstance(n, StrLit):
            return _quote(n.value)
        if isinstance(n, Name):
            return n.id
        if isinstance(n, Index):
            return f"{self._suffixed_atom(n.obj)}[{self.expr(n.key)}]"
        if isinstance(n, Dot):
            return f"{self._suffixed_atom(n.obj)}.{n.name}"
        if isinstance(n, Call):
            return f"{self._call_atom(n.func)}({self._args(n.args)})"
        if isinstance(n, Invoke):
            return f"{self._suffixed_atom(n.obj)}:{n.method}({self._args(n.args)})"
        if isinstance(n, FuncExpr):
            plist = ", ".join(n.params + (["..."] if n.is_vararg else []))
            hdr = f"function({plist})"
            # inline short empty bodies
            if not n.body:
                return f"{hdr} end"
            # multi-line function expression inside a table/list:
            sub = Printer()
            sub.indent = self.indent + 1
            for st in n.body:
                sub.emit_stmt(st)
            inner = "\n".join(sub.lines)
            return f"{hdr}\n{inner}\n{'    ' * self.indent}end"
        if isinstance(n, TableLit):
            return self._table(n)
        if isinstance(n, BinOp):
            return self._binop(n, parent_prec, side)
        if isinstance(n, UnOp):
            operand = self.expr(n.operand, _PREC["unary"], "left")
            if n.op == "not":
                return f"not {operand}"
            if isinstance(n.operand, UnOp) and n.operand.op == n.op and n.op in ("-", "~"):
                return f"{n.op}({operand})"   # prevent '--' comment fusion
            return f"{n.op}{operand}"
        return "[UNKNOWN]"  # never reached for valid ASTs

    _RIGHT_ASSOC = {"..", "^"}

    def _binop(self, n: BinOp, parent_prec: int, side: str) -> str:
        prec = _PREC.get(n.op, 0)
        s = self._binop_chain_render(n, prec)
        needs_parens = prec < parent_prec
        if prec == parent_prec:
            if n.op in self._RIGHT_ASSOC:
                needs_parens = side == "left"      # right-assoc: left child needs ()
            else:
                needs_parens = side == "right"     # left-assoc: right child needs ()
        return f"({s})" if needs_parens else s

    def _binop_chain_render(self, n: BinOp, prec: int) -> str:
        """Render a BinOp, flattening arbitrarily long same-precedence chains
        ITERATIVELY so obfuscation padding (thousands of +/-/.. terms) cannot
        overflow the call stack.  Mirrors the parenthesization the recursive
        form would produce:
          * chain lean matches associativity (left-assoc+left-lean,
            right-assoc+right-lean)  -> flat render, no inner parens;
          * lean opposes associativity (explicit parens in source) -> nested
            parens on inner links, exactly as the recursive printer emitted.
        """
        right_assoc = n.op in self._RIGHT_ASSOC

        def _same(x: Node) -> bool:
            return isinstance(x, BinOp) and _PREC.get(x.op, 0) == prec \
                and (x.op in self._RIGHT_ASSOC) == right_assoc

        left_chain = _same(n.left)
        right_chain = _same(n.right)
        if left_chain and right_chain:
            # mixed shape (both sides chains) -- balanced in practice; the
            # one-level recursion below is depth-safe (O(tree height)).
            left = self.expr(n.left, prec, "left")
            right = self.expr(n.right, prec, "right")
            return f"{left} {n.op} {right}"

        if left_chain and not right_assoc:
            # left-assoc, left-lean: flat, inner links need no parens
            items: List[Node] = [n.right]
            ops: List[str] = [n.op]
            cur = n.left
            while _same(cur):
                items.append(cur.right)
                ops.append(cur.op)
                cur = cur.left
            items.append(cur)
            items.reverse()
            parts = [self.expr(items[0], prec, "left")]
            for i in range(1, len(items)):
                parts.append(ops[i - 1])
                parts.append(self.expr(items[i], prec, "right"))
            return " ".join(parts)

        if right_chain and right_assoc:
            # right-assoc, right-lean: flat, inner links need no parens
            items = [n.left]
            ops = [n.op]
            cur = n.right
            while _same(cur):
                items.append(cur.left)
                ops.append(cur.op)
                cur = cur.right
            items.append(cur)
            parts = [self.expr(items[0], prec, "left")]
            for i in range(1, len(items)):
                parts.append(ops[i - 1])
                parts.append(self.expr(items[i], prec, "right"))
            return " ".join(parts)

        if left_chain and right_assoc:
            # right-assoc op, left-lean chain (our .. parser / explicit parens):
            # recursive form wrapped every left link in parens -- reproduce
            # iteratively with a running accumulator.
            items = [n.right]
            ops = [n.op]
            cur = n.left
            while _same(cur):
                items.append(cur.right)
                ops.append(cur.op)
                cur = cur.left
            items.append(cur)
            items.reverse()
            acc = self.expr(items[0], prec, "left")
            for i in range(1, len(items)):
                inner = f"{acc} {ops[i - 1]} {self.expr(items[i], prec, 'right')}"
                acc = f"({inner})" if i < len(items) - 1 else inner
            return acc

        if right_chain and not right_assoc:
            # left-assoc op, right-lean chain (explicit parens): mirror of
            # the above -- parens on inner links.
            items = [n.left]
            ops = [n.op]
            cur = n.right
            while _same(cur):
                items.append(cur.left)
                ops.append(cur.op)
                cur = cur.right
            items.append(cur)
            # association order: [items[0], n.op, ..., head]; render acc from
            # the head (rightmost) backwards with parens on inner links
            acc = self.expr(items[-1], prec, "right")
            for i in range(len(items) - 2, -1, -1):
                inner = f"{self.expr(items[i], prec, 'left')} {ops[i]} {acc}"
                acc = f"({inner})" if i > 0 else inner
            return acc

        left = self.expr(n.left, prec, "left")
        right = self.expr(n.right, prec, "right")
        return f"{left} {n.op} {right}"

    def _call_atom(self, f: Node) -> str:
        if isinstance(f, (Name, Dot, Index)):
            return self.expr(f)
        return f"({self.expr(f)})"

    def _suffixed_atom(self, n: Node) -> str:
        if isinstance(n, (Name, Dot, Index, Call, Invoke)):
            return self.expr(n)
        return f"({self.expr(n)})"

    def _args(self, args: List[Node]) -> str:
        return ", ".join(self.expr(a) for a in args)

    def _table(self, t: TableLit) -> str:
        if not t.entries:
            return "{}"
        flat = True
        for k, v in t.entries:
            if isinstance(v, FuncExpr) or isinstance(v, TableLit) and len(v.entries) > 4:
                flat = False
                break
            if k is not None and not isinstance(k, (StrLit, NumLit)):
                flat = False
                break
        if flat and len(t.entries) <= 8:
            parts = []
            for k, v in t.entries:
                if k is None:
                    parts.append(self.expr(v))
                else:
                    parts.append(f"[{self.expr(k)}] = {self.expr(v)}")
            return "{" + ", ".join(parts) + "}"
        # multi-line table
        lines = ["{"]
        self.indent += 1
        try:
            for k, v in t.entries:
                if k is None:
                    lines.append("    " * self.indent + self.expr(v) + ",")
                elif isinstance(k, StrLit) and re.fullmatch(r"[A-Za-z_]\w*", k.value or ""):
                    lines.append("    " * self.indent + f"{k.value} = {self.expr(v)},")
                else:
                    lines.append("    " * self.indent +
                                 f"[{self.expr(k)}] = {self.expr(v)},")
        finally:
            self.indent -= 1
        lines.append("    " * self.indent + "}")
        return "\n".join(lines)

    def result(self) -> str:
        return "\n".join(ln for ln in self.lines if ln is not None).rstrip() + "\n"


def _join_names(names: List[str]) -> str:
    return ", ".join(names)


def _fmt_num(v) -> str:
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 2**63:
            return str(int(v))
        r = repr(v)
        return r
    return str(v)


def _quote(s: str) -> str:
    return _quote_lua(s)
