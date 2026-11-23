"""Auth and small helpers shared by the routers."""
from __future__ import annotations

from fastapi import Request

from ..errors import BadRequest, Forbidden, SwitchboardError, Unauthorized
from ..ledger.budgets import ANONYMOUS, ApiKey


def gateway(request: Request):
    return request.app.state.gateway


def bearer(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else None


def authenticate(request: Request) -> ApiKey:
    gw = gateway(request)
    key = gw.keys.authenticate(bearer(request))
    if key is not None:
        return key
    if gw.settings.require_auth:
        raise Unauthorized("missing or invalid API key")
    return ANONYMOUS  # lite/dev mode: any token is accepted as the anonymous client


def require_admin(request: Request) -> None:
    gw = gateway(request)
    expected = gw.settings.admin_key
    if expected is None:
        if gw.settings.require_auth:
            raise Forbidden("admin endpoints are disabled: set ADMIN_KEY")
        return  # dev mode
    given = request.headers.get("x-admin-key") or bearer(request)
    if given != expected:
        raise Forbidden("invalid admin key")


async def read_json(request: Request) -> dict:
    limit = gateway(request).settings.max_body_bytes
    cl = request.headers.get("content-length")
    if cl and int(cl) > limit:
        raise SwitchboardErrorTooLarge()
    raw = await request.body()
    if len(raw) > limit:
        raise SwitchboardErrorTooLarge()
    import json
    try:
        return json.loads(raw or b"{}")
    except ValueError as e:
        raise BadRequest("invalid JSON body") from e


class SwitchboardErrorTooLarge(SwitchboardError):
    status, code, type = 413, "request_too_large", "invalid_request_error"
