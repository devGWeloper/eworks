"""IflowRetriever — iflow 컬렉션 도메인 검색"""

from langchain_core.documents import Document

from ._base_retriever import BaseRetriever


class IflowRetriever(BaseRetriever):

    _categories = {}

    async def search_iflow(self, query: str, k: int = 5) -> tuple[list, str]:
        # hybrid_search_with_score는 list[tuple[Document, float]] 반환
        response = await self.hybrid_search_with_score(query=query, k=k)
        return self._get_urls_contents(response)

    def _get_urls_contents(self, response_list: list[tuple[Document, float]]) -> tuple[list, str]:
        contents = ""
        iflow_urls = []
        cnt = 0
        for doc, score in response_list:
            # _parse_hits에서 text_field("text")는 page_content, 나머지는 metadata에 매핑됨
            iflow_content = doc.page_content
            nested_meta = doc.metadata.get("metadata") or {}
            iflow_id = nested_meta.get("doc_id", "")
            iflow_title = doc.metadata.get("title") or nested_meta.get("title", "")
            chunk_index = nested_meta.get("chunk_index", "")

            cnt += 1
            contents += f"title:{iflow_title}\n"
            contents += f"content:{iflow_content}\n"

            if iflow_id:
                iflow_urls.append(iflow_id)

        return iflow_urls, contents
