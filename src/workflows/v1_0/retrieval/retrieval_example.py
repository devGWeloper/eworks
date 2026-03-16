# 아래 로직들은 동작은 하지않는 논리적 코드
# 아래 로직들을 참고해서 현재 존재하는 retrieval 코어 함수 및 로직들을 수정하세요.

def hybrid_search(
        query,
        collection_name=config.COLLECTION_NAME,
        text_sparse_weight: float=0.1,
        text_dense_weight: float=0.9,
        limit=5
):_connect_milvus() #_connect_milvus는 milvus connection 맺는부분임. connection 맺는부분은 이미 우리 core에는 개발되어 있으므로 그것을 사용

    sparse_search_params={"metric_type":"BM25"}
    text_sparse_req = AnnSearchRequest(
        [query], "text_sparse", sparse_search_params, limit
    )

    #Dense 검색 요청
    dense_search_aprams={"metric_type":"COSINE", "params":{"ef":256}}

    query_vector = embedding_function.embed_query(query) # embedding_fuction은 OpenAIEmbeddings 객체임. 아마도 우리 core에는 개발 되어 있을거임
    text_dense_req = AnnSearchRequest([query_vector], "text_dense", dense_search_aprams, limit=limit)

    #가중치 기반 재순위화 (WeightedRanker)
    rerank = WeightedRanker(text_sparse_weight, text_dense_req)

    col = Collection(name=collection_name)

    # 하이브리드 검색 실행
    search_response = col.hybrid_search([text_sparse_req, text_dense_req],
                                        rerank = rerank,
                                        limit=limit,
                                        output_fields=["text", "title", "metadata"])[0] # 이게 아마 비즈니스 특화로직인것같은데.. 코어함수에서 이것도 cover가능하게 공통 기능으로 구현해줘야해

    return search_response