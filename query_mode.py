import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc

from utils.model_invocation import llm_model_func

if __name__ == "__main__":
    lightrag_isnt = LightRAG(
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM")),
            max_token_size=int(os.getenv("MAX_EMBEDDING_TOKEN_SIZE")),
            func=lambda texts: openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        ),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE"),
        addon_params={"language": os.getenv("SUMMARY_LANGUAGE")},
    )
    # query_par = QueryParam(mode="naive", only_need_context=True)
    # query_par = QueryParam(mode="local", only_need_context=True)
    # query_par = QueryParam(mode="global", only_need_context=True)
    # query_par = QueryParam(mode="hybrid", only_need_context=True)
    query_par = QueryParam(mode="mix", only_need_context=True)

    result = lightrag_isnt.query(
        query="describe table E-commerce Terminology Reference", param=query_par
    )
    print(result)