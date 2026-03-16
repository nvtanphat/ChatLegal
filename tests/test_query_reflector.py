from src.inference.query_reflector import QueryReflector


def test_extract_metadata_filter_article_and_doc_code() -> None:
    reflector = QueryReflector(llm=None, use_llm_reflect=False)
    metadata_filter = reflector.extract_metadata_filter("Dieu 124 Bo luat dan su 91/2015/QH13 quy dinh gi?")

    assert metadata_filter
    conditions = metadata_filter.get("$and", [])
    assert {"doc_code": {"$eq": "91/2015/QH13"}} in conditions
    article_condition = next(item for item in conditions if "$or" in item)
    assert {"article": {"$eq": "Dieu 124"}} in article_condition["$or"]
    assert {"article": {"$eq": "Điều 124"}} in article_condition["$or"]


def test_expand_queries_heuristic_generates_variants() -> None:
    reflector = QueryReflector(llm=None, use_llm_reflect=False)
    variants = reflector.expand_queries("Dieu 124 BLDS quy dinh gi?")

    assert variants
    assert any("dieu 124" in item.lower() for item in variants)


def test_extract_metadata_filter_none_for_generic_question() -> None:
    reflector = QueryReflector(llm=None, use_llm_reflect=False)
    metadata_filter = reflector.extract_metadata_filter("Hop dong mua ban nha can luu y gi?")

    assert metadata_filter is None
