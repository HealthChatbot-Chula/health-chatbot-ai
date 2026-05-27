from __future__ import annotations

import time
import functools
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

_local = threading.local()


def _get_stack() -> list["_Span"]:
    """Stack ของ span ที่กำลัง active อยู่ (ยังไม่จบ)"""
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def _get_roots() -> list["_Span"]:
    """เก็บ span ระดับ root ตามลำดับเวลา"""
    if not hasattr(_local, "roots"):
        _local.roots = []
    return _local.roots


def _clear() -> None:
    _local.stack = []
    _local.roots = []


@dataclass
class _Span:
    name: str
    elapsed_ms: float = 0.0
    children: list["_Span"] = field(default_factory=list)
    input_tokens: int = 0      
    output_tokens: int = 0


@contextmanager
def trace(name: str, **extra):
    span = _Span(name=name)
    stack = _get_stack()

    # ถ้ามี parent อยู่ใน stack ให้เป็น child ของมัน
    if stack:
        stack[-1].children.append(span)
    else:
        _get_roots().append(span)

    stack.append(span)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        span.elapsed_ms = (time.perf_counter() - t0) * 1000
        stack.pop()


def trace_fn(name: str | None = None):
    def decorator(func: Callable) -> Callable:
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span = _Span(name=label)
            stack = _get_stack()

            if stack:
                stack[-1].children.append(span)
            else:
                _get_roots().append(span)

            stack.append(span)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                span.elapsed_ms = (time.perf_counter() - t0) * 1000
                stack.pop()
                # parent จบแล้ว → print children ก่อน แล้วค่อย print ตัวเอง
                for child in span.children:
                    _print_inline(child, indent=True)
                _print_inline(span, indent=False)
            return result

        return wrapper
    return decorator

def record_tokens(name_or_span, response) -> None:
    meta = getattr(response, "usage_metadata", None) or {}
    input_tok = meta.get("input_tokens", 0) or meta.get("prompt_token_count", 0)
    output_tok = meta.get("output_tokens", 0) or meta.get("candidates_token_count", 0)
    
    stack = _get_stack()
    if stack:
        stack[-1].input_tokens += input_tok
        stack[-1].output_tokens += output_tok


_BAR   = 36
_WARN  = 500
_ERROR = 2000
_R = "\033[0m"; _G = "\033[32m"; _Y = "\033[33m"; _RED = "\033[31m"
_B = "\033[1m"; _D = "\033[2m";  _C = "\033[36m"


def _col(ms: float) -> str:
    return _RED if ms >= _ERROR else _Y if ms >= _WARN else _G


def _bar(ms: float, max_ms: float) -> str:
    n = int(min(ms / max(max_ms, 1), 1.0) * _BAR)
    return "█" * n + "░" * (_BAR - n)


def _print_inline(s: _Span, indent: bool) -> None:
    c = _col(s.elapsed_ms)
    prefix = "  " if not indent else ""
    tok_info = ""
    if s.input_tokens or s.output_tokens:
        tok_info = f"  {_D}[in:{s.input_tokens} out:{s.output_tokens}]{_R}"
    print(f"{_C}⏱  {_R}{prefix}{s.name:<48}{c}{s.elapsed_ms:>8.1f} ms{_R}{tok_info}")


class LatencyReport:
    def __init__(self, label: str = "request"):
        self.label = label
        self._t0 = time.perf_counter()
        _clear()

    @classmethod
    def begin(cls, label: str = "request") -> "LatencyReport":
        return cls(label)

    def _flatten(self, spans: list[_Span], result: list, depth: int = 0):
        for s in spans:
            result.append((s, depth))
            self._flatten(s.children, result, depth + 1)

    def print(self) -> None:
        roots = _get_roots()
        total_ms = (time.perf_counter() - self._t0) * 1000
        all_spans = []
        self._flatten(roots, all_spans)
        max_ms = max((s.elapsed_ms for s, _ in all_spans), default=1)
        sep = "─" * 72

        print(f"\n{_B}{sep}{_R}")
        print(f"{_B}  LATENCY REPORT  ›  {self.label}{_R}")
        print(sep)

        for s, depth in all_spans:
            c = _col(s.elapsed_ms)
            pct = s.elapsed_ms / total_ms * 100
            bar = _bar(s.elapsed_ms, max_ms)
            indent_str = "    " * depth
            prefix = "└─ " if depth > 0 else ""
            tok_str = ""
            if s.input_tokens or s.output_tokens:
                tok_str = f"  {_D}in:{s.input_tokens:>5} out:{s.output_tokens:>4}{_R}"
            print(f"  {indent_str}{prefix}{s.name:<44} {c}{s.elapsed_ms:>7.1f} ms{_R}{tok_str}  {_D}{bar} {pct:4.1f}%{_R}")

        llm_ms = sum(s.elapsed_ms for s, _ in all_spans if "LLM" in s.name)
        total_in = sum(s.input_tokens for s, _ in all_spans) 
        total_out = sum(s.output_tokens for s, _ in all_spans)
        print(sep)
        print(f"  {_B}Total wall time  {_R}{_col(total_ms)}{total_ms:>8.1f} ms{_R}")
        if llm_ms:
            print(f"  {_B}LLM calls only  {_R}{_col(llm_ms)}{llm_ms:>8.1f} ms{_R}  {_D}({llm_ms/total_ms*100:.0f}% of total){_R}")
        if total_in or total_out:                              
            print(f"  {_B}Tokens total     {_R}in:{total_in}  out:{total_out}  total:{total_in + total_out}")
        print(f"{_B}{sep}{_R}\n")
        _clear()