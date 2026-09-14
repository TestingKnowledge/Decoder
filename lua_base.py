"""
lua_base.py — safe Lua/Luau value model, operational limits, and lexer.

Shared foundation for engine_ast.py (parser/printer) and deobfuscator.py
(analysis engine). No code from obfuscated samples is ever executed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config import ENGINE


class EngineTimeout(Exception):
    """Raised when the analysis exceeds its time budget."""


class EngineOverflow(Exception):
    """Raised when a resource limit (nodes / ops / string size) is hit."""


class _Limits:
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.fold_ops = 0
        self.emulated_steps = 0
        self.loops_emulated = 0
        self.string_built = 0
        self.nodes = 0

    def tick_op(self):
        self.fold_ops += 1
        if self.fold_ops > ENGINE["max_fold_ops"]:
            raise EngineOverflow("constant-folding operation limit exceeded")
        self.check_time()

    def tick_step(self):
        self.emulated_steps += 1
        if self.emulated_steps > ENGINE["max_total_emulated_steps"]:
            raise EngineOverflow("emulated step limit exceeded")
        if self.emulated_steps % 2048 == 0:
            self.check_time()

    def tick_node(self):
        self.nodes += 1
        if self.nodes > ENGINE["max_ast_nodes"]:
            raise EngineOverflow("AST node limit exceeded")
        if self.nodes % 8192 == 0:
            self.check_time()

    def tick_string(self, size: int):
        self.string_built += size
        if self.string_built > ENGINE["max_string_build"]:
            raise EngineOverflow("string-build limit exceeded")

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise EngineTimeout()


# ══════════════════════════════════════════════════════════════════════════════
# §1  LUA VALUES — safe emulation of Lua numeric & string semantics
# ══════════════════════════════════════════════════════════════════════════════

class LuaTable:
    """Emulated Lua table (list + hash parts) used for constant folding."""
    __slots__ = ("list", "hash")

    def __init__(self):
        self.list: List[Any] = []
        self.hash: Dict[Any, Any] = {}

    def set(self, key, value):
        if isinstance(key, float) and key == int(key) and 1 <= key <= 2**31:
            k = int(key)
            while len(self.list) < k:
                self.list.append(None)
            self.list[k - 1] = value
        elif isinstance(key, (int, float)) and isinstance(key, float) is False \
                and 1 <= key <= 2**31 and float(key).is_integer():
            k = int(key)
            while len(self.list) < k:
                self.list.append(None)
            self.list[k - 1] = value
        else:
            self.hash[key] = value

    def get(self, key):
        if isinstance(key, (int, float)):
            k = int(key)
            if k == key and 1 <= k <= len(self.list):
                return self.list[k - 1]
            return self.hash.get(key if isinstance(key, float) is False else float(key))
        return self.hash.get(key)

    def __repr__(self):
        return f"table({len(self.list)} list, {len(self.hash)} hash)"


def _to_int32(v: int) -> int:
    """Wrap to signed 32-bit (Lua 5.2+/Luau bitwise semantics)."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _toint(x: float) -> Optional[int]:
    """Lua 5.3+ 'math.toint': integer iff representable."""
    if isinstance(x, float) and x.is_integer() and abs(x) < 2**63:
        return int(x)
    return None


class LuaValue:
    """A concrete Lua value produced by constant folding.

    KINDS: 'nil', 'bool', 'num', 'str', 'table'
    """
    __slots__ = ("kind", "num", "str", "bval", "tbl")

    def __init__(self, kind: str, num=None, string=None, bval=None, tbl=None):
        self.kind = kind
        self.num = num
        self.str = string
        self.bval = bval
        self.tbl = tbl

    # ── constructors ────────────────────────────────────────────────────────
    @staticmethod
    def nil() -> "LuaValue":
        return LuaValue("nil")

    @staticmethod
    def boolean(b: bool) -> "LuaValue":
        return LuaValue("bool", bval=b)

    @staticmethod
    def number(n) -> "LuaValue":
        return LuaValue("num", num=float(n) if not isinstance(n, int) else n)

    @staticmethod
    def string(s: str) -> "LuaValue":
        return LuaValue("str", string=s)

    @staticmethod
    def table(t: LuaTable) -> "LuaValue":
        return LuaValue("table", tbl=t)

    # ── type checks ─────────────────────────────────────────────────────────
    def truthy(self) -> bool:
        return not (self.kind == "nil" or (self.kind == "bool" and self.bval is False))

    def is_num(self) -> bool:
        return self.kind == "num"

    def is_int(self) -> bool:
        return (self.kind == "num" and
                (isinstance(self.num, int) or
                 (isinstance(self.num, float) and self.num.is_integer())))

    def int_value(self) -> Optional[int]:
        if self.is_int():
            return int(self.num)
        return None

    # ── operators ───────────────────────────────────────────────────────────
    def binop(self, op: str, other: "LuaValue", lim: _Limits) -> "LuaValue":
        lim.tick_op()
        a, b = self, other

        # string coercion (Lua 5.1/5.2 & Luau arithmetic)
        if op in ("+", "-", "*", "/", "%", "^"):
            a = a._coerce_num(op)
            b = b._coerce_num(op)
            if a is None or b is None:
                return LuaValue.nil()  # non-foldable → symbol
        else:
            if a.kind == "str" or b.kind == "str":
                if op == "..":
                    return a._concat(b, lim)
                return LuaValue.nil()  # non-foldable → symbol

        an, bn = a.num, b.num

        if op in ("&", "|", "~", "<<", ">>"):
            ia, ib = a.int_value(), b.int_value()
            if ia is None or ib is None:
                return LuaValue.nil()
            if op == "&":
                r = ia & ib
            elif op == "|":
                r = ia | ib
            elif op == "~":
                r = ia ^ ib
            elif op == "<<":
                if ib < 0 or ib >= 64:
                    return LuaValue.nil()
                r = ia << ib
            else:
                if ib < 0 or ib >= 64:
                    return LuaValue.nil()
                r = ia >> ib
            return LuaValue.number(_to_int32(r))

        try:
            if op == "+":
                r = an + bn
            elif op == "-":
                r = an - bn
            elif op == "*":
                r = an * bn
            elif op == "/":
                if bn == 0:
                    return LuaValue.nil()  # inf → not a finite constant we fold
                r = an / bn
                return LuaValue.number(r)
            elif op == "%":
                if bn == 0:
                    return LuaValue.nil()
                # Lua: a - floor(a/b)*b
                r = an - math.floor(an / bn) * bn
            elif op == "^":
                r = an ** bn
            elif op == "..":
                return a._concat(b, lim)
            else:
                return LuaValue.nil()
        except (OverflowError, ZeroDivisionError, ValueError):
            return LuaValue.nil()
        if isinstance(r, float) and (math.isinf(r) or math.isnan(r)):
            return LuaValue.nil()
        return LuaValue.number(r)

    def _concat(self, b: "LuaValue", lim: _Limits) -> "LuaValue":
        sa = self._coerce_str()
        sb = b._coerce_str()
        if sa is None or sb is None:
            return LuaValue.nil()
        lim.tick_string(len(sa) + len(sb))
        return LuaValue.string(sa + sb)

    def unop(self, op: str, lim: _Limits) -> "LuaValue":
        lim.tick_op()
        if op == "-":
            n = self._coerce_num(op)
            return LuaValue.nil() if n is None else LuaValue.number(-n.num)
        if op == "not":
            return LuaValue.boolean(not self.truthy())
        if op == "#":
            if self.kind == "str":
                return LuaValue.number(len(self.str))
            if self.kind == "table":
                return LuaValue.number(len(self.tbl.list))
        if op == "~":
            i = self.int_value()
            return LuaValue.nil() if i is None else LuaValue.number(_to_int32(~i))
        return LuaValue.nil()

    def _coerce_num(self, op) -> Optional["LuaValue"]:
        if self.kind == "num":
            return self
        if self.kind == "str" and self.str is not None:
            try:
                s = self.str.strip()
                if re.fullmatch(r"-?(?:0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)", s):
                    n = float(int(s, 16)) if re.fullmatch(r"-?0[xX][0-9a-fA-F]+", s) else float(s)
                    return LuaValue.number(n)
            except (ValueError, OverflowError):
                pass
        return None

    def _coerce_str(self) -> Optional[str]:
        if self.kind == "str":
            return self.str
        if self.kind == "num":
            if isinstance(self.num, int):
                return str(self.num)
            if self.num is not None and self.num.is_integer() and abs(self.num) < 2**63:
                return str(int(self.num))
            return None
        return None

    # ── rendering ───────────────────────────────────────────────────────────
    def render(self) -> str:
        if self.kind == "nil":
            return "nil"
        if self.kind == "bool":
            return "true" if self.bval else "false"
        if self.kind == "num":
            return _fmt_number(self.num)
        if self.kind == "str":
            return _quote_lua(self.str)
        return "table"

    def render_shows(self) -> str:
        """Compact display used by the analysis report."""
        if self.kind == "str":
            s = self.str
            if len(s) > 96:
                s = s[:93] + "..."
            return f"{_quote_lua(s)} ({len(self.str)} bytes)"
        if self.kind == "table":
            t = self.tbl
            items = [v.render_shows() if v is not None else "nil"
                     for v in t.list[:ENGINE["analysis_head_rows"]]]
            extra = len(t.list) - len(items)
            txt = "{" + ", ".join(items)
            if extra > 0:
                txt += f", +{extra} more"
            if t.hash:
                txt += ", hash keys: " + ", ".join(str(k) for k in list(t.hash)[:8])
            return txt + "}"
        return self.render()

    def __repr__(self):
        return f"<LuaValue {self.render()}>"


def _fmt_number(n) -> str:
    if isinstance(n, float):
        if n.is_integer() and abs(n) < 2**63:
            return str(int(n))
        return repr(n)
    return str(n)


def _quote_lua(s: str) -> str:
    """Render a Python str as a Lua double-quoted literal."""
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif 32 <= o < 127:
            out.append(ch)
        elif o < 256:
            out.append(f"\\{o}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)




# ══════════════════════════════════════════════════════════════════════════════
# §2  LEXER — Lua 5.1 → 5.4 + Luau dialect
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tok:
    type: str            # 'name' | 'number' | 'string' | 'keyword' | 'op' | 'eof'
    value: Any
    pos: int             # byte offset in source
    line: int
    col: int
    raw: str = ""


_KEYWORDS = {
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
    "goto", "if", "in", "local", "nil", "not", "or", "repeat", "return",
    "then", "true", "until", "while", "continue",  # 'continue' = Luau
}

_LONGOPS = {
    "...", "..", "::", "<<", ">>", "//", "==", "~=", "<=", ">=",
    # Luau compound assignment (2-char forms; "..=" and "//=" are 3-char)
    "+=", "-=", "*=", "/=", "%=", "^=",
}

_THREEOPS = {"...", "===", "!==", "..=", "//="}

_HEX_DIGITS = set("0123456789abcdefABCDEF")


def lex(src: str, lim: _Limits) -> List[Tok]:
    """Tokenize Lua/Luau source. Raises SyntaxError on malformed input."""
    toks: List[Tok] = []
    i, n = 0, len(src)
    line, bol = 1, 0          # bol = beginning of current line
    pending_lbrackets: List[Tuple[int, int, int, int]] = []  # (count, i, line, col)

    def col_of(pos: int) -> int:
        return pos - bol + 1

    while i < n:
        lim.tick_node()
        c = src[i]

        # ── whitespace / newlines ───────────────────────────────────────────
        if c == "\n":
            i += 1
            line += 1
            bol = i
            continue
        if c in " \t\r\v\f":
            i += 1
            continue

        # ── short comment ───────────────────────────────────────────────────
        if src.startswith("--", i):
            j = i + 2
            # long comment?  --[[ ... ]] or --[=[ ... ]=]
            m = re.match(r"--\[(=*)\[", src[i:])
            if m:
                closer = "]" + "=" * len(m.group(1)) + "]"
                end = src.find(closer, i + m.end())
                if end == -1:
                    raise SyntaxError(f"line {line}: unfinished long comment")
                seg = src[i:end + len(closer)]
                nl = seg.count("\n")
                line += nl
                if nl:
                    bol = i + seg.rindex("\n") + 1
                i = end + len(closer)
                continue
            # plain line comment
            end = src.find("\n", j)
            if end == -1:
                end = n
            i = end
            continue

        # ── names / keywords ────────────────────────────────────────────────
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            ttype = "keyword" if word in _KEYWORDS else "name"
            toks.append(Tok(ttype, word, i, line, col_of(i), word))
            i = j
            continue

        # ── numbers ─────────────────────────────────────────────────────────
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            if src.startswith("0x", i) or src.startswith("0X", i):
                j = i + 2
                start = j
                while j < n and src[j] in _HEX_DIGITS:
                    j += 1
                # hex fraction / exponent (Lua 5.2+)
                if j < n and src[j] == ".":
                    j += 1
                    while j < n and src[j] in _HEX_DIGITS:
                        j += 1
                if j < n and src[j] in "pP":
                    j += 1
                    if j < n and src[j] in "+-":
                        j += 1
                    while j < n and src[j].isdigit():
                        j += 1
                raw = src[i:j]
                if j == start and "." not in raw[2:]:
                    raise SyntaxError(f"line {line}: malformed hex number")
                val = _parse_lua_number(raw)
                toks.append(Tok("number", val, i, line, col_of(i), raw))
            else:
                while j < n and src[j].isdigit():
                    j += 1
                if j < n and src[j] == ".":
                    j += 1
                    while j < n and src[j].isdigit():
                        j += 1
                if j < n and src[j] in "eE":
                    k = j + 1
                    if k < n and src[k] in "+-":
                        k += 1
                    if k < n and src[k].isdigit():
                        j = k
                        while j < n and src[j].isdigit():
                            j += 1
                raw = src[i:j]
                val = _parse_lua_number(raw)
                toks.append(Tok("number", val, i, line, col_of(i), raw))
            if len(toks) > ENGINE["max_tokens"]:
                raise EngineOverflow("token limit exceeded")
            i = j
            continue

        # ── long bracket strings / pending [==[ ─────────────────────────────
        if c == "[":
            m = re.match(r"\[(=*)\[", src[i:])
            if m:
                closer = "]" + "=" * len(m.group(1)) + "]"
                end = src.find(closer, i + m.end())
                if end == -1:
                    raise SyntaxError(f"line {line}: unfinished long string")
                body = src[i + m.end():end]
                nl = body.count("\n")
                start_line = line
                if body.startswith("\n"):
                    body = body[1:]
                elif body.startswith("\r\n"):
                    body = body[2:]
                toks.append(Tok("string", body, i, start_line, col_of(i), src[i:end + len(closer)]))
                line += nl
                if nl:
                    bol = i + m.end() + src[i + m.end():end].rindex("\n") + 1
                i = end + len(closer)
                continue
            toks.append(Tok("op", "[", i, line, col_of(i), "["))
            i += 1
            continue

        # ── quoted strings ──────────────────────────────────────────────────
        if c in ("\"", "'"):
            quote = c
            j = i + 1
            buf: List[str] = []
            while j < n:
                ch = src[j]
                if ch == "\\":
                    if j + 1 >= n:
                        raise SyntaxError(f"line {line}: unfinished string")
                    e = src[j + 1]
                    if e == "n":
                        buf.append("\n"); j += 2
                    elif e == "t":
                        buf.append("\t"); j += 2
                    elif e == "r":
                        buf.append("\r"); j += 2
                    elif e == "a":
                        buf.append("\a"); j += 2
                    elif e == "b":
                        buf.append("\b"); j += 2
                    elif e == "f":
                        buf.append("\f"); j += 2
                    elif e == "v":
                        buf.append("\v"); j += 2
                    elif e == "\\":
                        buf.append("\\"); j += 2
                    elif e == '"':
                        buf.append('"'); j += 2
                    elif e == "'":
                        buf.append("'"); j += 2
                    elif e == "\n":
                        buf.append("\n"); j += 2
                    elif e.isdigit():
                        k = j + 1
                        ds = ""
                        while k < n and len(ds) < 3 and src[k].isdigit():
                            ds += src[k]; k += 1
                            if int(ds) > 255:
                                break
                        try:
                            buf.append(chr(int(ds)))
                        except (ValueError, OverflowError):
                            pass
                        j = k
                    elif e == "x":
                        k = j + 2
                        hs = ""
                        while k < n and len(hs) < 2 and src[k] in _HEX_DIGITS:
                            hs += src[k]; k += 1
                        if hs:
                            buf.append(chr(int(hs, 16)))
                            j = k
                        else:
                            raise SyntaxError(f"line {line}: bad \\x escape")
                    elif e == "z":
                        k = j + 2
                        while k < n and src[k] in " \t\r\n":
                            k += 1
                        j = k
                    elif e == "u":
                        mo = re.match(r"\{([0-9a-fA-F]+)\}", src[j + 2:])
                        if mo:
                            try:
                                buf.append(chr(int(mo.group(1), 16)))
                            except (ValueError, OverflowError):
                                raise SyntaxError(f"line {line}: bad \\u escape")
                            j = j + 2 + mo.end()
                        else:
                            raise SyntaxError(f"line {line}: bad \\u escape")
                    else:
                        raise SyntaxError(f"line {line}: invalid escape \\{e}")
                    continue
                if ch == quote:
                    break
                if ch == "\n":
                    raise SyntaxError(f"line {line}: unfinished string")
                buf.append(ch)
                j += 1
            if j >= n:
                raise SyntaxError(f"line {line}: unfinished string")
            s = "".join(buf)
            lim.tick_string(len(s))
            toks.append(Tok("string", s, i, line, col_of(i), src[i:j + 1]))
            i = j + 1
            continue

        # ── operators ───────────────────────────────────────────────────────
        three = src[i:i + 3]
        two = src[i:i + 2]
        if three in _THREEOPS:
            toks.append(Tok("op", three, i, line, col_of(i), three))
            i += 3
            continue
        if two in _LONGOPS:
            toks.append(Tok("op", two, i, line, col_of(i), two))
            i += 2
            continue
        if c in "+-*/%^#&|~<>=(){}];:,.?":  # note: '[' handled above
            toks.append(Tok("op", c, i, line, col_of(i), c))
            i += 1
            continue
        # unknown byte
        raise SyntaxError(f"line {line}: unexpected symbol {c!r}")

    toks.append(Tok("eof", None, n, line, col_of(n), ""))
    return toks


def _parse_lua_number(raw: str) -> float:
    """Parse a Lua numeric literal (int stays int for exact bitwise math)."""
    try:
        low = raw.lower()
        if low.startswith("0x"):
            body = low[2:].replace("_", "")
            if "p" in body:
                return float.fromhex(body)
            if "." in body:
                ipart, fpart = body.split(".", 1)
                val = int(ipart or "0", 16)
                frac = 0.0
                scale = 1 / 16
                for h in fpart:
                    frac += int(h, 16) * scale
                    scale /= 16
                return val + frac
            return int(body, 16)
        clean = raw.replace("_", "")
        if re.fullmatch(r"-?\d+", clean):
            return int(clean)
        return float(clean)
    except (ValueError, OverflowError):
        raise SyntaxError(f"malformed number {raw!r}")
