import json
import unittest

from switchboard.errors import ProviderAuthError, ProviderBadRequest, ProviderUnavailable, RateLimited
from switchboard.providers.openai_compat import error_for_status, parse_completion, parse_retry_after, parse_sse_data
from switchboard.types import ChatRequest, Chunk


class TestParsing(unittest.TestCase):
    def test_retry_after_seconds_and_date(self):
        self.assertEqual(parse_retry_after("12"), 12.0)
        self.assertEqual(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412480 - 5), 5.0)
        self.assertIsNone(parse_retry_after("garbage"))
        self.assertIsNone(parse_retry_after(None))

    def test_status_mapping(self):
        e = error_for_status(429, "slow down", 3.0)
        self.assertIsInstance(e, RateLimited)
        self.assertEqual(e.retry_after, 3.0)
        self.assertIsInstance(error_for_status(401, "", None), ProviderAuthError)
        self.assertIsInstance(error_for_status(400, "", None), ProviderBadRequest)
        self.assertIsInstance(error_for_status(502, "", None), ProviderUnavailable)

    def test_usage_real_vs_estimated(self):
        req = ChatRequest("m", [{"role": "user", "content": "hello there"}])
        real = parse_completion({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 7, "completion_tokens": 2}}, req)
        self.assertFalse(real.usage.estimated)
        self.assertEqual(real.usage.prompt_tokens, 7)
        est = parse_completion({"choices": [{"message": {"content": "hi"}}]}, req)
        self.assertTrue(est.usage.estimated)

    def test_sse_parsing(self):
        self.assertIsNone(parse_sse_data("[DONE]"))
        c = parse_sse_data(json.dumps({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}))
        self.assertEqual(c.content, "Hi")
        u = parse_sse_data(json.dumps({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}))
        self.assertEqual(u.usage.completion_tokens, 2)

    def test_chunk_schema(self):
        d = Chunk(content="x").to_openai("id1", "m", 1, first=True)
        self.assertEqual(d["object"], "chat.completion.chunk")
        self.assertEqual(d["choices"][0]["delta"], {"role": "assistant", "content": "x"})


if __name__ == "__main__":
    unittest.main()
