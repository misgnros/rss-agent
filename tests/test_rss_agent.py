"""
Unit tests for rss_agent_cli.py

Covers:
  - clean_html()
  - extract_keywords()
  - init_db() / get_db_connection()
  - preprocess_article()
  - generate_comment()
  - generate_rebuttal()
  - process_article()
  - build_graph()
"""
import asyncio
import os
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set a fake API key before the module import triggers genai.configure()
os.environ.setdefault("GOOGLE_API_KEY", "fake-test-key")

import rss_agent_cli as rac
from rss_agent_cli import (
    ArticleState,
    build_graph,
    clean_html,
    extract_keywords,
    generate_comment,
    generate_rebuttal,
    get_db_connection,
    init_db,
    preprocess_article,
    process_article,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_state(**overrides) -> ArticleState:
    """Return an ArticleState dict with sensible defaults, overridden by kwargs."""
    base: ArticleState = {
        "url": "https://example.com/article",
        "title": "Test Article Title",
        "summary": "This is a test article summary with enough content.",
        "full_content": "",
        "cleaned_content": "This is a test article summary with enough content.",
        "char_count": 51,
        "keywords": ["test", "article"],
        "should_process": True,
        "comments": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# clean_html
# ---------------------------------------------------------------------------

class TestCleanHtml:
    def test_empty_string(self):
        assert clean_html("") == ""

    def test_none_returns_empty(self):
        assert clean_html(None) == ""

    def test_plain_text_unchanged(self):
        assert clean_html("hello world") == "hello world"

    def test_removes_simple_tag(self):
        assert clean_html("<p>Hello</p>") == "Hello"

    def test_removes_nested_tags(self):
        assert clean_html("<div><p><strong>Bold</strong></p></div>") == "Bold"

    def test_unescapes_amp(self):
        assert clean_html("A &amp; B") == "A & B"

    def test_unescapes_lt_gt(self):
        # &lt;tag&gt; is first unescaped to <tag>, which is then treated as
        # an HTML tag and removed — leaving an empty string.
        assert clean_html("&lt;tag&gt;") == ""

    def test_collapses_whitespace(self):
        assert clean_html("hello   world\n\t!") == "hello world !"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_html("  hello  ") == "hello"

    def test_mixed_html_and_entities(self):
        assert clean_html("<p>Hello &amp; World</p>") == "Hello & World"

    def test_preserves_text_content(self):
        result = clean_html("<h1>Title</h1><p>Body text here.</p>")
        assert "Title" in result
        assert "Body text here." in result
        assert "<" not in result


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_empty_string(self):
        assert extract_keywords("") == []

    def test_none_returns_empty(self):
        assert extract_keywords(None) == []

    def test_max_keywords_limit(self):
        text = "alpha beta gamma delta epsilon zeta"
        assert len(extract_keywords(text, max_keywords=3)) <= 3

    def test_default_max_keywords(self):
        words = ["apple", "banana", "cherry", "mango", "peach", "grape", "melon"]
        text = " ".join(words * 2)
        assert len(extract_keywords(text)) <= 5

    def test_stopwords_excluded(self):
        # All words are in the stopwords set
        result = extract_keywords("the is at which on and or but")
        assert result == []

    def test_short_words_excluded(self):
        # Words shorter than 4 chars are not matched by r'\b[a-zA-Z]{4,}\b'
        result = extract_keywords("cat dog foo bar baz")
        assert result == []

    def test_most_frequent_word_first(self):
        text = "python python python java java ruby"
        result = extract_keywords(text, max_keywords=1)
        assert result == ["python"]

    def test_case_insensitive(self):
        text = "Python PYTHON python Java JAVA"
        result = extract_keywords(text, max_keywords=2)
        assert "python" in result
        assert "java" in result

    def test_returns_list(self):
        assert isinstance(extract_keywords("hello world test data"), list)


# ---------------------------------------------------------------------------
# init_db / get_db_connection
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_db_creates_articles_table(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        with patch("rss_agent_cli.DB_FILE", db_file):
            init_db()
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
        )
        assert cur.fetchone() is not None
        conn.close()

    def test_init_db_creates_expected_columns(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        with patch("rss_agent_cli.DB_FILE", db_file):
            init_db()
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(articles)")
        columns = {row[1] for row in cur.fetchall()}
        assert columns == {"id", "url", "title", "summary", "full_content", "char_count"}
        conn.close()

    def test_init_db_is_idempotent(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        with patch("rss_agent_cli.DB_FILE", db_file):
            init_db()
            init_db()  # should not raise

    def test_get_db_connection_yields_connection(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        with patch("rss_agent_cli.DB_FILE", db_file):
            with get_db_connection() as conn:
                assert isinstance(conn, sqlite3.Connection)

    def test_get_db_connection_closes_after_block(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        captured = []
        with patch("rss_agent_cli.DB_FILE", db_file):
            with get_db_connection() as conn:
                captured.append(conn)
        # After exiting the context manager the connection is closed
        import contextlib
        with contextlib.suppress(Exception):
            # A closed connection raises ProgrammingError on cursor()
            captured[0].cursor()
            pytest.fail("Expected an error on closed connection")


# ---------------------------------------------------------------------------
# preprocess_article
# ---------------------------------------------------------------------------

class TestPreprocessArticle:
    async def test_valid_article_should_process(self):
        state = make_state(
            summary="<p>This article contains enough content to be processed.</p>"
        )
        result = await preprocess_article(state)
        assert result["should_process"] is True

    async def test_html_stripped_from_cleaned_content(self):
        state = make_state(summary="<h1>Title</h1><p>Body content here.</p>")
        result = await preprocess_article(state)
        assert "<h1>" not in result["cleaned_content"]
        assert "<p>" not in result["cleaned_content"]

    async def test_char_count_reflects_cleaned_length(self):
        plain = "A" * 50
        state = make_state(summary=plain)
        result = await preprocess_article(state)
        assert result["char_count"] == 50

    async def test_keywords_are_extracted(self):
        state = make_state(
            summary="Python programming language Python Python code development"
        )
        result = await preprocess_article(state)
        assert "python" in result["keywords"]

    async def test_short_content_sets_should_process_false(self):
        # Less than 10 chars after cleaning
        state = make_state(summary="Hi")
        result = await preprocess_article(state)
        assert result["should_process"] is False

    async def test_empty_url_sets_should_process_false(self):
        state = make_state(
            url="",
            summary="This article has enough content to normally pass the length check.",
        )
        result = await preprocess_article(state)
        assert result["should_process"] is False

    async def test_url_without_scheme_sets_should_process_false(self):
        state = make_state(
            url="not-a-valid-url",
            summary="This article has enough content to normally pass the length check.",
        )
        result = await preprocess_article(state)
        assert result["should_process"] is False

    async def test_returns_expected_keys(self):
        state = make_state()
        result = await preprocess_article(state)
        assert set(result.keys()) == {
            "full_content",
            "cleaned_content",
            "char_count",
            "keywords",
            "should_process",
        }


# ---------------------------------------------------------------------------
# generate_comment
# ---------------------------------------------------------------------------

class TestGenerateComment:
    async def test_returns_response_text(self):
        mock_response = MagicMock()
        mock_response.text = "テストコメントです。"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model):
            result = await generate_comment(make_state(), "TestAgent", "コメント")

        assert result == "テストコメントです。"

    async def test_strips_whitespace(self):
        mock_response = MagicMock()
        mock_response.text = "  前後の空白  "
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model):
            result = await generate_comment(make_state(), "Agent", "instruction")

        assert result == "前後の空白"

    async def test_empty_response_returns_error_string(self):
        mock_response = MagicMock()
        mock_response.text = None
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await generate_comment(make_state(), "Agent", "instruction")

        assert result.startswith("[Error:")

    async def test_api_exception_returns_error_string(self):
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(side_effect=Exception("API Error"))

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await generate_comment(make_state(), "Agent", "instruction")

        assert "[Error:" in result

    async def test_retries_on_failure(self):
        """Verify the function retries max_retries times before giving up."""
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(side_effect=Exception("fail"))

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await generate_comment(make_state(), "Agent", "instruction")

        # 3 retries expected (max_retries=3)
        assert mock_model.generate_content_async.call_count == 3


# ---------------------------------------------------------------------------
# generate_rebuttal
# ---------------------------------------------------------------------------

class TestGenerateRebuttal:
    async def test_returns_rebuttal_text(self):
        mock_response = MagicMock()
        mock_response.text = "反論テキストです。"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model):
            result = await generate_rebuttal(
                make_state(), "AgentA", "AgentB", "AgentBのコメント", "反論してください。"
            )

        assert result == "反論テキストです。"

    async def test_exception_returns_error_string(self):
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(side_effect=RuntimeError("fail"))

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await generate_rebuttal(
                make_state(), "AgentA", "AgentB", "comment", "instruction"
            )

        assert "[Error:" in result

    async def test_retries_on_failure(self):
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(side_effect=Exception("fail"))

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            await generate_rebuttal(
                make_state(), "AgentA", "AgentB", "comment", "instruction"
            )

        assert mock_model.generate_content_async.call_count == 3

    async def test_empty_target_comment_still_calls_api(self):
        mock_response = MagicMock()
        mock_response.text = "応答"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model):
            result = await generate_rebuttal(
                make_state(), "AgentA", "AgentB", "", "instruction"
            )

        assert result == "応答"


# ---------------------------------------------------------------------------
# process_article
# ---------------------------------------------------------------------------

class TestProcessArticle:
    async def test_skip_when_should_not_process(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        with patch("rss_agent_cli.DB_FILE", db_file):
            init_db()
            result = await process_article(make_state(should_process=False))
        assert result["comments"] == {}

    async def test_new_article_saved_to_db(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        mock_response = MagicMock()
        mock_response.text = "テストコメント"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.DB_FILE", db_file), \
             patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("rss_agent_cli.AGENTS", [("Agent1", "コメント")]), \
             patch("rss_agent_cli.ENABLE_REBUTTALS", False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            init_db()
            await process_article(make_state())

        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "SELECT url FROM articles WHERE url=?", ("https://example.com/article",)
        )
        assert cur.fetchone() is not None
        conn.close()

    async def test_duplicate_article_not_reinserted(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        mock_response = MagicMock()
        mock_response.text = "テストコメント"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.DB_FILE", db_file), \
             patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("rss_agent_cli.AGENTS", [("Agent1", "コメント")]), \
             patch("rss_agent_cli.ENABLE_REBUTTALS", False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            init_db()
            state = make_state()
            await process_article(state)
            await process_article(state)  # same URL — second call is a duplicate

        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM articles WHERE url=?",
            ("https://example.com/article",),
        )
        assert cur.fetchone()[0] == 1
        conn.close()

    async def test_comments_generated_for_each_agent(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        mock_response = MagicMock()
        mock_response.text = "コメント"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        agents = [("AgentA", "指示A"), ("AgentB", "指示B")]

        with patch("rss_agent_cli.DB_FILE", db_file), \
             patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("rss_agent_cli.AGENTS", agents), \
             patch("rss_agent_cli.ENABLE_REBUTTALS", False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            init_db()
            result = await process_article(make_state())

        assert "AgentA" in result["comments"]
        assert "AgentB" in result["comments"]

    async def test_rebuttals_generated_when_enabled(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        mock_response = MagicMock()
        mock_response.text = "コメントまたは反論"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        agents = [("AgentA", "指示A"), ("AgentB", "指示B")]

        with patch("rss_agent_cli.DB_FILE", db_file), \
             patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("rss_agent_cli.AGENTS", agents), \
             patch("rss_agent_cli.ENABLE_REBUTTALS", True), \
             patch("rss_agent_cli.MAX_REBUTTALS_PER_AGENT", 1), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            init_db()
            result = await process_article(make_state())

        for agent_name in ["AgentA", "AgentB"]:
            entry = result["comments"][agent_name]
            assert isinstance(entry, dict)
            assert "main" in entry
            assert "rebuttals" in entry

    async def test_rebuttals_target_other_agents(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        mock_response = MagicMock()
        mock_response.text = "応答テキスト"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        agents = [("AgentA", "指示A"), ("AgentB", "指示B")]

        with patch("rss_agent_cli.DB_FILE", db_file), \
             patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("rss_agent_cli.AGENTS", agents), \
             patch("rss_agent_cli.ENABLE_REBUTTALS", True), \
             patch("rss_agent_cli.MAX_REBUTTALS_PER_AGENT", 1), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            init_db()
            result = await process_article(make_state())

        # AgentA should have rebutted AgentB and vice-versa
        assert "AgentB" in result["comments"]["AgentA"]["rebuttals"]
        assert "AgentA" in result["comments"]["AgentB"]["rebuttals"]

    async def test_comments_dict_normalized_when_rebuttals_disabled(self, tmp_path):
        """Even without rebuttals, comments are normalized to {main, rebuttals} dicts."""
        db_file = str(tmp_path / "test.db")
        mock_response = MagicMock()
        mock_response.text = "コメント"
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("rss_agent_cli.DB_FILE", db_file), \
             patch("rss_agent_cli.genai.GenerativeModel", return_value=mock_model), \
             patch("rss_agent_cli.AGENTS", [("AgentA", "指示A")]), \
             patch("rss_agent_cli.ENABLE_REBUTTALS", False), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            init_db()
            result = await process_article(make_state())

        entry = result["comments"]["AgentA"]
        assert isinstance(entry, dict)
        assert entry["main"] == "コメント"
        assert entry["rebuttals"] == {}


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_graph_contains_preprocess_node(self):
        graph = build_graph()
        assert "preprocess_article" in graph.nodes

    def test_graph_contains_process_node(self):
        graph = build_graph()
        assert "process_article" in graph.nodes

    def test_graph_compiles_without_error(self):
        compiled = build_graph().compile()
        assert compiled is not None

    def test_build_graph_returns_state_graph(self):
        from langgraph.graph import StateGraph
        graph = build_graph()
        assert isinstance(graph, StateGraph)
