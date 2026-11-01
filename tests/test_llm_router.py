"""llm-router 离线单元测试（不打真实端点）。"""
import importlib
import json
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import llm_router  # noqa: E402


def _reset_state():
    """每个用例前重置模块级可变状态，避免用例间污染。"""
    importlib.reload(llm_router)


class TestCandidateBuilders(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_yaml_multi_key_expansion(self):
        cfg = {
            "endpoints": [
                {"name": "a", "priority": 1, "api_url": "http://a/v1",
                 "api_key": ["k1", "k2"], "model": "m1"},
            ]
        }
        cands = llm_router._build_candidates_from_yaml(cfg)
        self.assertEqual([c.tag for c in cands], ["P1-k1", "P1-k2"])
        self.assertEqual([c.key for c in cands], ["k1", "k2"])

    def test_yaml_dedup_same_triple(self):
        cfg = {
            "endpoints": [
                {"name": "a", "priority": 1, "api_url": "http://a/v1", "api_key": "k", "model": "m"},
                {"name": "b", "priority": 2, "api_url": "http://a/v1", "api_key": "k", "model": "m"},
            ]
        }
        cands = llm_router._build_candidates_from_yaml(cfg)
        self.assertEqual(len(cands), 1)

    def test_yaml_sorted_by_priority(self):
        cfg = {
            "endpoints": [
                {"name": "c", "priority": 3, "api_url": "http://c/v1", "api_key": "k3", "model": "m"},
                {"name": "a", "priority": 1, "api_url": "http://a/v1", "api_key": "k1", "model": "m"},
                {"name": "b", "priority": 2, "api_url": "http://b/v1", "api_key": "k2", "model": "m"},
            ]
        }
        cands = llm_router._build_candidates_from_yaml(cfg)
        self.assertEqual([c.url for c in cands], ["http://a/v1", "http://b/v1", "http://c/v1"])

    def test_yaml_env_interpolation(self):
        cfg = {
            "endpoints": [
                {"name": "a", "priority": 1, "api_url": "${MY_URL}",
                 "api_key": "${MY_KEY}", "model": "${MY_MODEL:def-m}"},
            ]
        }
        with mock.patch.dict(os.environ, {"MY_URL": "http://env/v1", "MY_KEY": "envkey"}):
            cands = llm_router._build_candidates_from_yaml(cfg)
        self.assertEqual(cands[0].url, "http://env/v1")
        self.assertEqual(cands[0].key, "envkey")
        self.assertEqual(cands[0].model, "def-m")  # MY_MODEL 未设 → 用默认值

    def test_yaml_missing_key_skipped(self):
        cfg = {
            "endpoints": [
                {"name": "a", "priority": 1, "api_url": "http://a/v1", "model": "m"},  # 无 key
                {"name": "b", "priority": 2, "api_url": "http://b/v1", "api_key": "k2", "model": "m"},
            ]
        }
        cands = llm_router._build_candidates_from_yaml(cfg)
        self.assertEqual([c.url for c in cands], ["http://b/v1"])

    def test_env_builder_p3_no_break_on_missing_key(self):
        env = {
            "LLM_P3_URL": "http://p3/v1", "LLM_P3_KEY": "k3",
            "LLM_P4_URL": "http://p4/v1",  # P4 有 URL 无 KEY → 跳过不中断
            "LLM_P5_URL": "http://p5/v1", "LLM_P5_KEY": "k5",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cands = llm_router._build_candidates_env()
        self.assertEqual([c.tag for c in cands], ["P3", "P5"])

    def test_env_builder_multi_key_pool(self):
        env = {"LLM_P3_URL": "http://p3/v1", "LLM_P3_KEY1": "ka", "LLM_P3_KEY2": "kb"}
        with mock.patch.dict(os.environ, env, clear=False):
            cands = llm_router._build_candidates_env()
        self.assertEqual([c.key for c in cands], ["ka", "kb"])
        self.assertEqual([c.tag for c in cands], ["P3", "P3-k2"])


class TestRefusalDetection(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_refusal_prefix(self):
        self.assertTrue(llm_router._looks_like_refusal("[advisor] 无法回答"))

    def test_refusal_phrase(self):
        self.assertTrue(llm_router._looks_like_refusal("As an AI, I cannot help with that."))

    def test_normal_text(self):
        self.assertFalse(llm_router._looks_like_refusal("磁盘使用率 92%，建议清理日志目录。"))

    def test_empty(self):
        self.assertFalse(llm_router._looks_like_refusal(""))


class TestErrorClassification(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_connection_error_is_hard(self):
        e = llm_router.requests.exceptions.ConnectionError("connection refused")
        self.assertTrue(llm_router._is_hard_error(e))

    def test_401_is_hard(self):
        resp = mock.Mock()
        resp.status_code = 401
        e = llm_router.requests.exceptions.HTTPError("401", response=resp)
        self.assertTrue(llm_router._is_hard_error(e))

    def test_429_is_rate_limited_not_hard(self):
        resp = mock.Mock()
        resp.status_code = 429
        e = llm_router.requests.exceptions.HTTPError("429", response=resp)
        self.assertTrue(llm_router._is_rate_limited(e))
        self.assertFalse(llm_router._is_hard_error(e))

    def test_timeout_is_rate_limited(self):
        e = llm_router.requests.exceptions.Timeout("read timed out")
        self.assertTrue(llm_router._is_rate_limited(e))
        self.assertFalse(llm_router._is_hard_error(e))


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        _reset_state()
        self.assertTrue(llm_router.LLM_HEALTH_ENABLED)

    def test_mark_dead_then_skip(self):
        url = "http://cb-test/v1"
        self.assertFalse(llm_router._is_dead(url))
        llm_router._mark_dead(url)
        self.assertTrue(llm_router._is_dead(url))

    def test_mark_good_recovers(self):
        url = "http://cb-recover/v1"
        llm_router._mark_dead(url)
        llm_router._mark_good(url, latency_ms=120)
        self.assertFalse(llm_router._is_dead(url))
        self.assertEqual(llm_router._h(url).last_latency, 120)

    def test_fail_count_accumulates(self):
        url = "http://cb-count/v1"
        llm_router._mark_dead(url)
        llm_router._mark_dead(url)
        self.assertEqual(llm_router._h(url).fail_count, 2)


class TestPromptCache(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_set_get_roundtrip(self):
        key = llm_router._prompt_cache_key("m", "s", "p")
        llm_router._cache_set(key, "答案")
        self.assertEqual(llm_router._cache_get(key), "答案")

    def test_miss_returns_none(self):
        self.assertIsNone(llm_router._cache_get("nonexistent"))

    def test_eviction_over_limit(self):
        for i in range(510):
            llm_router._cache_set(f"k{i}", "v")
        self.assertLessEqual(len(llm_router._PROMPT_CACHE), 510)


class TestSortByHealth(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_live_first_sorted_by_latency(self):
        cands = [
            llm_router.Candidate("http://slow/v1", "k", "m", "P2"),
            llm_router.Candidate("http://fast/v1", "k", "m", "P1"),
        ]
        llm_router._mark_good("http://slow/v1", latency_ms=900)
        llm_router._mark_good("http://fast/v1", latency_ms=100)
        order = llm_router._sort_by_health(cands)
        self.assertEqual(order[0].url, "http://fast/v1")

    def test_dead_sorted_last(self):
        cands = [
            llm_router.Candidate("http://dead/v1", "k", "m", "P1"),
            llm_router.Candidate("http://alive/v1", "k", "m", "P2"),
        ]
        llm_router._mark_dead("http://dead/v1")
        llm_router._mark_good("http://alive/v1", latency_ms=500)
        order = llm_router._sort_by_health(cands)
        self.assertEqual(order[0].url, "http://alive/v1")


class TestReasoningBudget(unittest.TestCase):
    """推理预算耗尽 → 放大 max_tokens 重试一次。"""

    def setUp(self):
        _reset_state()

    def _fake_response(self, content, ct, rt, finish="length"):
        msg = {"content": content}
        if rt is not None:
            msg["reasoning_content"] = "thinking..."
        return {
            "choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"completion_tokens": ct,
                      "completion_tokens_details": {"reasoning_tokens": rt}},
        }

    def test_budget_exhausted_retry_with_bigger_budget(self):
        first = self._fake_response("", 2000, 2000)
        second = self._fake_response("最终答案", 2203, 2000, finish="stop")
        resp1, resp2 = mock.Mock(), mock.Mock()
        resp1.raise_for_status.return_value = None
        resp1.json.return_value = first
        resp2.raise_for_status.return_value = None
        resp2.json.return_value = second

        with mock.patch.object(llm_router, "_get_session") as gs:
            gs.return_value.post.side_effect = [resp1, resp2]
            choice, msg, content = llm_router._post_with_reasoning_budget(
                "http://x/v1/chat/completions", {}, {"max_tokens": 2000}, 30, True)
        self.assertEqual(content, "最终答案")
        # 第二次请求的 max_tokens 应被放大到 CEILING
        _, kwargs = gs.return_value.post.call_args
        self.assertEqual(kwargs["json"]["max_tokens"], llm_router.REASONING_BUDGET_CEILING)

    def test_not_exhausted_no_retry(self):
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = self._fake_response("正常答案", 100, 0, finish="stop")
        with mock.patch.object(llm_router, "_get_session") as gs:
            gs.return_value.post.return_value = resp
            _choice, _msg, content = llm_router._post_with_reasoning_budget(
                "http://x/v1/chat/completions", {}, {"max_tokens": 2000}, 30, True)
        self.assertEqual(content, "正常答案")
        self.assertEqual(gs.return_value.post.call_count, 1)


class TestCallLlmRouting(unittest.TestCase):
    """call_llm 端到端路由行为（mock HTTP）。"""

    def setUp(self):
        _reset_state()

    def _cfg(self):
        return {
            "defaults": {"model": "m"},
            "endpoints": [
                {"name": "a", "priority": 1, "api_url": "http://a/v1", "api_key": "ka", "model": "m"},
                {"name": "b", "priority": 2, "api_url": "http://b/v1", "api_key": "kb", "model": "m"},
            ],
        }

    def test_first_healthy_endpoint_wins(self):
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        with mock.patch.object(llm_router, "_load_yaml_config", return_value=self._cfg()), \
             mock.patch.object(llm_router, "_get_session") as gs:
            gs.return_value.post.return_value = resp
            out = llm_router.call_llm("hi")
        self.assertEqual(out, "ok")
        gs.return_value.post.assert_called_once()
        self.assertIn("http://a/v1", gs.return_value.post.call_args.args[0])

    def test_fallback_on_hard_error(self):
        resp_ok = mock.Mock()
        resp_ok.raise_for_status.return_value = None
        resp_ok.json.return_value = {
            "choices": [{"message": {"content": "from-b"}, "finish_reason": "stop"}]
        }
        with mock.patch.object(llm_router, "_load_yaml_config", return_value=self._cfg()), \
             mock.patch.object(llm_router, "_get_session") as gs:
            gs.return_value.post.side_effect = [
                llm_router.requests.exceptions.ConnectionError("connection refused"),
                resp_ok,
            ]
            out = llm_router.call_llm("hi")
        self.assertEqual(out, "from-b")
        self.assertEqual(gs.return_value.post.call_count, 2)

    def test_all_fail_returns_none(self):
        with mock.patch.object(llm_router, "_load_yaml_config", return_value=self._cfg()), \
             mock.patch.object(llm_router, "_get_session") as gs, \
             mock.patch.object(llm_router.time, "sleep"):
            gs.return_value.post.side_effect = (
                llm_router.requests.exceptions.Timeout("t"))
            out = llm_router.call_llm("hi")
        self.assertIsNone(out)

    def test_no_config_returns_none(self):
        with mock.patch.object(llm_router, "_load_yaml_config", return_value=None), \
             mock.patch.object(llm_router, "_build_candidates_env", return_value=[]):
            self.assertIsNone(llm_router.call_llm("hi"))

    def test_empty_prompt_returns_none(self):
        self.assertIsNone(llm_router.call_llm(""))


class TestCallLlmJson(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_valid_tool_call_parsed(self):
        fake = {
            "content": "",
            "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "emit",
                             "arguments": json.dumps({"severity": "P0", "impact": "全站"})},
            }],
            "used_fallback": False, "model": "m",
        }
        with mock.patch.object(llm_router, "call_llm_with_tools", return_value=fake):
            out = llm_router.call_llm_json("提取", schema={"type": "object", "properties": {}})
        self.assertEqual(out, {"severity": "P0", "impact": "全站"})

    def test_no_tool_call_returns_none(self):
        fake = {"content": "我无法回答", "tool_calls": [], "used_fallback": True, "model": "m"}
        with mock.patch.object(llm_router, "call_llm_with_tools", return_value=fake):
            out = llm_router.call_llm_json("提取", schema={"type": "object", "properties": {}})
        self.assertIsNone(out)

    def test_empty_schema_raises(self):
        with self.assertRaises(ValueError):
            llm_router.call_llm_json("x", schema={})

    def test_empty_prompt_returns_none(self):
        self.assertIsNone(llm_router.call_llm_json("", schema={"type": "object"}))


class TestListEndpointsMasked(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_key_masked(self):
        cfg = {"endpoints": [
            {"name": "a", "priority": 1, "api_url": "http://a/v1",
             "api_key": "sk-verylongsecretkey-1234567890", "model": "m"},
        ]}
        with mock.patch.object(llm_router, "_load_yaml_config", return_value=cfg):
            eps = llm_router.list_endpoints()
        k = eps[0]["key"]
        self.assertNotIn("verylongsecretkey", k)  # 明文不出现在诊断视图
        self.assertIn("...", k)


class TestEnvShParsing(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def test_parse_export_lines(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                         encoding="utf-8") as fh:
            fh.write('# comment\nexport A_URL="http://a/v1"\n'
                     "export B_KEY='bk'\nexport EMPTY=\n")
            path = fh.name
        try:
            with mock.patch.dict(os.environ, {"LLM_ENV_FILE": path}, clear=False):
                llm_router._ENV_SH_CACHE.clear()
                data = llm_router._load_env_sh()
            self.assertEqual(data.get("A_URL"), "http://a/v1")
            self.assertEqual(data.get("B_KEY"), "bk")
            self.assertNotIn("EMPTY", data)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
