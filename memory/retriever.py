from sentence_transformers import SentenceTransformer
import torch
from collections import OrderedDict


# 전역 캐시 (LRU 방식)
embedding_cache = OrderedDict()
CACHE_SIZE = 100  # 캐싱할 최대 임베딩 개수


class Retriever:
    """
    Sentence-BERT를 기반으로 문장 임베딩을 생성하는 클래스.
    또한 key와 query 임베딩 간의 유사도 계산을 통해 상위 결과를 반환하는 search_in_embeds 메서드를 제공합니다.
    """
    def __init__(self, device='cpu', model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.device = device
        self.embedder = SentenceTransformer(model_name, device=device)

    def embed(self, texts):
        """
        주어진 문자열 리스트를 임베딩 텐서로 변환합니다.
        :param texts: list[str]
        :return: torch.Tensor, shape: (num_texts, embed_dim)
        """
        embeddings = self.embedder.encode(texts, convert_to_tensor=True, device=self.device)
        return embeddings

    @torch.no_grad()
    def search_in_embeds(self, key_embeds, query_embeds, topk: int = None, similarity_threshold: float = None,
                         return_embeds: bool = False, return_scores: bool = False):

        """
        key_embeds와 query_embeds 간의 유사도를 계산하여,
        topk 개의 가장 유사한 결과(또는 similarity_threshold 이상의 결과)를 반환합니다.
        """
        if int(topk is None) + int(similarity_threshold is None) != 1:
            raise ValueError("You should specify either topk or similarity_threshold but not both!")

        # 마지막 두 차원을 전치하도록 수정
        scores = query_embeds @ key_embeds.transpose(-1, -2)  # shape: (M, N)
        batch_request = len(query_embeds.shape) > 1
        if not batch_request:
            scores = scores.reshape(1, -1)
        num_q = scores.shape[0]

        if topk is not None:
            sorted_idx = scores.argsort(dim=-1, descending=True)
            selected_idx = sorted_idx[:, :topk]
            selected_idx = selected_idx.tolist()
        else:
            selected_idx = [[] for _ in range(num_q)]
            nonzero_indices = (scores >= similarity_threshold).nonzero(as_tuple=False)
            for item in nonzero_indices:
                q_id, k_id = item[0].item(), item[1].item()
                selected_idx[q_id].append(k_id)

        result = {'idx': selected_idx}
        if return_embeds:
            result['embeds'] = [
                [key_embeds[k_id] for k_id in selected_idx[q_id]]
                for q_id in range(num_q)
            ]
        if return_scores:
            result['scores'] = [
                [scores[q_id, k_id].item() for k_id in selected_idx[q_id]]
                for q_id in range(num_q)
            ]
        if not batch_request:
            result = {k: v[0] for k, v in result.items()}
        return result





def get_cached_embeddings(texts, retriever):
    """
    주어진 문자열 리스트에 대해 캐시된 임베딩이 있으면 사용하고,
    없는 경우 Retriever.embed()를 통해 계산한 후 캐싱합니다.
    :param texts: 임베딩을 구할 문자열 리스트
    :param retriever: Retriever 인스턴스
    :return: (num_texts, embed_dim) 크기의 텐서
    """
    results = [None] * len(texts)
    texts_to_compute = []
    indices_to_compute = []

    for i, text in enumerate(texts):
        if text in embedding_cache:
            embedding_cache.move_to_end(text)
            results[i] = embedding_cache[text]
        else:
            texts_to_compute.append(text)
            indices_to_compute.append(i)

    if texts_to_compute:
        computed = retriever.embed(texts_to_compute)
        for idx, text, emb in zip(indices_to_compute, texts_to_compute, computed):
            results[idx] = emb
            embedding_cache[text] = emb
            if len(embedding_cache) > CACHE_SIZE:
                embedding_cache.popitem(last=False)

    return torch.stack(results)


"""
@torch.no_grad()
def graph_retr_search(start_query, triplets, retriever, max_depth: int = 2,
                      topk: int = 3, post_retrieve_threshold: float = 0.7,
                      verbose: int = 2):
    # triplets 임베딩 (2차원 텐서: (N, embed_dim))
    key_embeds = get_cached_embeddings(triplets, retriever)

    current_level = [start_query]
    depth = {start_query: 0}
    result = []
    visited = set([start_query])

    while current_level:
        query_embeds = get_cached_embeddings(current_level, retriever)
        # query_embeds가 1차원일 경우 2차원으로 변환
        if query_embeds.ndim == 1:
            query_embeds = query_embeds.unsqueeze(0)

        # 마지막 두 차원만 전치하여 (M, N) 형태의 유사도 행렬 계산
        scores = query_embeds @ key_embeds.transpose(-1, -2)
        effective_topk = min(topk, scores.shape[1])
        topk_values, topk_indices = scores.topk(effective_topk, dim=1)

        next_level = []
        for i, query in enumerate(current_level):
            for score, idx in zip(topk_values[i].tolist(), topk_indices[i].tolist()):
                if score < post_retrieve_threshold:
                    continue
                triplet = triplets[idx]
                parts = [part.strip() for part in triplet.split(",")]
                if len(parts) < 3:
                    continue
                v1, relation, v2 = parts[:3]
                for v in [v1, v2]:
                    if v not in visited and depth[query] < max_depth:
                        visited.add(v)
                        next_level.append(v)
                        depth[v] = depth[query] + 1
                if triplet not in result:
                    result.append(triplet)
                if verbose >= 2:
                    print(f"[Depth {depth[query]}] Query: '{query}' -> Triplet: '{triplet}' (Score: {score:.3f})")
        current_level = next_level
    return result
"""

@torch.no_grad()
def graph_retr_search(start_triplet, triplets, retriever, max_depth: int = 2,
                      topk: int = 3, post_retrieve_threshold: float = 0.7,
                      verbose: int = 2):
    """
    시작 쿼리(triplet)를 기반으로 주어진 triplets에서 BFS 방식으로 관련 결과를 탐색합니다.
    쿼리와 후보 모두 "v1, relation, v2" 형식의 문자열이며, 임베딩을 통해 전체 문맥을 비교합니다.

    :param start_triplet: 시작 트리플릿 문자열 (예: "user, is in, operations conference barracks")
    :param triplets: 후보 트리플릿 문자열들의 리스트
    :param retriever: Retriever 인스턴스
    :param max_depth: 최대 탐색 깊이 (현재 트리플릿에서 확장할 수 있는 단계)
    :param topk: 각 쿼리마다 고려할 상위 k개 후보
    :param post_retrieve_threshold: 유사도 임계치 (이 값 미만인 후보는 무시)
    :param verbose: 디버그 출력을 위한 출력 레벨 (2 이상이면 상세 출력)
    :return: 조건에 맞는 트리플릿 문자열들의 리스트
    """
    # 후보 트리플릿 임베딩 (2차원 텐서: (N, embed_dim))
    key_embeds = get_cached_embeddings(triplets, retriever)

    current_level = [start_triplet]  # 검색 쿼리(트리플릿) 리스트
    depth = {start_triplet: 0}         # 각 트리플릿의 탐색 깊이 기록
    result = set()                   # 최종 검색된 트리플릿 집합
    visited = set([start_triplet])   # 중복 검색 방지를 위한 방문 집합

    while current_level:
        query_embeds = get_cached_embeddings(current_level, retriever)
        if query_embeds.ndim == 1:
            query_embeds = query_embeds.unsqueeze(0)
        # 안전하게 마지막 두 차원을 전치하여 유사도 행렬 계산
        scores = query_embeds @ key_embeds.transpose(-1, -2)
        effective_topk = min(topk, scores.shape[1])
        topk_values, topk_indices = scores.topk(effective_topk, dim=1)

        next_level = []
        for i, query in enumerate(current_level):
            current_depth = depth[query]
            for score, idx in zip(topk_values[i].tolist(), topk_indices[i].tolist()):
                if score < post_retrieve_threshold:
                    continue
                candidate_triplet = triplets[idx]
                if candidate_triplet in visited:
                    continue
                # 후보 트리플릿을 결과에 추가하고, 아직 깊이 제한 안에 있다면 확장 대상으로 포함
                result.add(candidate_triplet)
                if current_depth < max_depth:
                    next_level.append(candidate_triplet)
                    depth[candidate_triplet] = current_depth + 1
                    visited.add(candidate_triplet)
                if verbose >= 2:
                    print(f"[Depth {current_depth}] Query: '{query}' -> Triplet: '{candidate_triplet}' (Score: {score:.3f})")
        current_level = next_level
    return list(result)




# Example usage:
if __name__ == "__main__":
    # Retriever 인스턴스 초기화 (예: GPU 사용 시 device='cuda')
    my_retriever = Retriever(device='cpu')

    # 예시 triplets (각 문자열은 "v1, relation, v2" 형식)
    triplets = [
        "apple, is a, fruit",
        "fruit, can be, eaten",
        "banana, is a, fruit",
        "apple, is related to, banana",
        "orange, is a, fruit",
    ]

    start_query = "apple"

    results = graph_retr_search(start_query, triplets, my_retriever,
                                max_depth=3, topk=3, post_retrieve_threshold=0.55, verbose=1)

    print("\nGraph Search Results:")
    for r in results:
        print(r)
