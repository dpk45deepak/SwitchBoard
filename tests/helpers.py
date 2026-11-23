from switchboard.config import Settings
from switchboard.clock import Clock
from switchboard.db.session import Database
from switchboard.gateway import build_gateway
from switchboard.providers.mock import MockBehavior, MockProvider
from switchboard.providers.registry import Registry
from switchboard.types import ChatRequest


def mk_gateway(clock=None, db=None, behavior=None, behavior2=None, **settings_kw):
    clock = clock or Clock()
    st = Settings(db_path=":memory:", embedder="hashing", **settings_kw)
    reg = Registry.load(settings=st)
    fast = lambda: MockBehavior(latency_ms={"cheap": 1, "balanced": 1, "strong": 1})  # noqa: E731
    providers = {"mock": MockProvider("mock", clock, 1, behavior or fast()),
                 "mock2": MockProvider("mock2", clock, 2, behavior2 or fast())}
    gw = build_gateway(st, clock, db or Database(), providers, registry=reg)
    return gw


def chat(text="What is the capital of France?", model="mock-strong", **kw):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "temperature": 0}
    body.update(kw)
    return ChatRequest.from_openai(body)
