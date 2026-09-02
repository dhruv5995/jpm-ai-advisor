from jpm_ai_advisor.tools.knowledge_search import KnowledgeStore


def test_search_finds_relevant_chunk() -> None:
    store = KnowledgeStore()
    hits = store.search("balanced risk band 3 allocation")
    assert hits
    assert any("Balanced" in hit.heading for hit in hits)


def test_search_as_text_formats_hits_with_doc_and_heading() -> None:
    store = KnowledgeStore()
    text = store.search_as_text("REITs inflation hedge correlation to equities")
    assert "asset_classes" in text
    assert "Real Estate" in text


def test_search_as_text_handles_no_match() -> None:
    store = KnowledgeStore()
    text = store.search_as_text("zzzzz qqqqq nonexistent gibberish")
    assert text == "No relevant results in the knowledge base."
