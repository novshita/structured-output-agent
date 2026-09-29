"""Example tools plus Pydantic models for their responses."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .schemas import CurrencyNormalization

_CURRENCY_ALIASES = {
    "$": "USD", "us$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "₹": "INR", "inr": "INR", "rs": "INR", "rs.": "INR", "rupee": "INR", "rupees": "INR",
    "c$": "CAD", "cad": "CAD", "a$": "AUD", "aud": "AUD",
    "¥": "JPY", "jpy": "JPY", "yen": "JPY", "円": "JPY", "万円": "JPY",
    "zł": "PLN", "pln": "PLN", "clp": "CLP",
    "chf": "CHF", "sgd": "SGD", "s$": "SGD", "aed": "AED", "dirham": "AED",
}


def normalize_currency(raw: str) -> dict:
    """Map a currency symbol or name ("$", "rupees", "€") to an ISO 4217 code."""
    code = _CURRENCY_ALIASES.get(raw.strip().lower())
    return {"raw": raw, "code": code, "recognized": code is not None}


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., Any]
    response_model: type[BaseModel]


TOOLS: dict[str, Tool] = {
    "normalize_currency": Tool(
        name="normalize_currency",
        description="Convert a currency symbol or name found in the posting to an ISO 4217 code.",
        parameters={"raw": "the currency symbol or word exactly as it appears, e.g. '$' or 'rupees'"},
        fn=normalize_currency,
        response_model=CurrencyNormalization,
    ),
}


class ToolError(Exception):
    """A tool call was malformed or the tool returned data that failed validation."""

    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type


def run_tool(name: str, arguments: dict, tools: dict[str, Tool] = TOOLS) -> BaseModel:
    """Execute a tool and validate its response. Raises ToolError on any failure."""
    tool = tools.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name!r}; available: {sorted(tools)}", "tool_response_invalid")
    try:
        raw = tool.fn(**arguments)
    except TypeError as e:
        raise ToolError(f"bad arguments for {name}: {e}", "tool_response_invalid") from e
    try:
        return tool.response_model.model_validate(raw)
    except ValidationError as e:
        raise ToolError(f"{name} returned invalid data: {e}", "tool_response_invalid") from e


def describe_tools(tools: dict[str, Tool] = TOOLS) -> str:
    lines = []
    for t in tools.values():
        params = ", ".join(f'"{k}": <{v}>' for k, v in t.parameters.items())
        lines.append(f"- {t.name}({{{params}}}): {t.description}")
    return "\n".join(lines)
