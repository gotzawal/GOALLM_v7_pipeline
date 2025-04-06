import os
import re
import ast
import json
import torch
import numpy as np
from copy import deepcopy
from inspect import signature
import matplotlib.pyplot as plt


def clear_triplet(triplet):
    # "I" -> inventory, "P" -> user, "N" -> npc
    if triplet[0] == "I":
        triplet = ("inventory", triplet[1], triplet[2])
    if triplet[1] == "I":
        triplet = (triplet[0], "inventory", triplet[2])
    if triplet[0] == "P":
        triplet = ("user", triplet[1], triplet[2])
    if triplet[1] == "P":
        triplet = (triplet[0], "user", triplet[2])
    if triplet[0] == "N":
        triplet = ("npc", triplet[1], triplet[2])
    if triplet[1] == "N":
        triplet = (triplet[0], "npc", triplet[2])
    return [triplet[0].lower().strip('''"'. `;:'''),
            triplet[1].lower().strip('''"'. `;:'''),
            {"label": triplet[2]["label"].lower().strip('''"'. `;:''')}]

def sort_scores(data):
    """
    data['idx']와 data['scores']가 단일 값일 경우에도 리스트 형태로 다룰 수 있도록 보정합니다.
    각 쿼리에 대해 인덱스와 점수를 인덱스 기준 오름차순으로 정렬합니다.
    """
    idx_data = data['idx']
    score_data = data['scores']

    # 단일 값이면 리스트로 감싸기
    if not isinstance(idx_data, list):
        idx_data = [idx_data]
    if not isinstance(score_data, list):
        score_data = [score_data]
    # 만약 내부가 단일 int/float라면 한 번 더 리스트로 감싸기
    if idx_data and not isinstance(idx_data[0], list):
        idx_data = [idx_data]
    if score_data and not isinstance(score_data[0], list):
        score_data = [score_data]

    sorted_idx_all = []
    sorted_scores_all = []
    for idx_list, score_list in zip(idx_data, score_data):
        paired = list(zip(idx_list, score_list))
        paired_sorted = sorted(paired, key=lambda x: x[0])
        if paired_sorted:
            idx_sorted, score_sorted = zip(*paired_sorted)
        else:
            idx_sorted, score_sorted = [], []
        sorted_idx_all.append(list(idx_sorted))
        sorted_scores_all.append(list(score_sorted))
    return {'idx': sorted_idx_all, 'scores': sorted_scores_all}


def process_triplets(raw_triplets):
    raw_triplets = raw_triplets.split(";")
    triplets = []
    for triplet in raw_triplets:
        if len(triplet.split(",")) != 3:
            continue
        if triplet[0] in "123456789":
            triplet = triplet[2:]
        subj, relation, obj = triplet.split(",")
        subj = subj.split(":")[-1].strip(''' '\n"''')
        relation = relation.strip(''' '\n"''')
        obj = obj.strip(''' '\n"''')
        if len(subj) == 0 or len(relation) == 0 or len(obj) == 0:
            continue
        triplets.append([subj, obj, {"label": relation}])
    return triplets


def parse_triplets_removing(text):
    text = text.split("[[")[-1] if "[[" in text else text.split("[\n[")[-1]
    text = text.replace("[", "")
    text = text.strip("]")
    pairs = text.split("],")
    parsed_triplets = []
    for pair in pairs:
        splitted_pair = pair.split("->")
        if len(splitted_pair) != 2:
            continue
        first_triplet = splitted_pair[0].split(",")
        if len(first_triplet) != 3:
            continue
        subj = first_triplet[0].strip(''' '"\n''')
        rel = first_triplet[1].strip(''' '"\n''')
        obj = first_triplet[2].strip(''' '"\n''')
        parsed_triplets.append([subj, obj, {"label": rel}])
    return parsed_triplets


def find_top_episodic_emb(A, B, obs_plan_embedding, retriever):
    results = {}
    if not B:
        return results
    # B 딕셔너리의 각 value[1]을 스택하여 (N, embed_dim) 형태의 2차원 텐서를 만듭니다.
    key_embeddings = torch.stack([value[1] for value in B.values()])

    # obs_plan_embedding이 1차원인 경우 2차원으로 변경
    if obs_plan_embedding.ndim == 1:
        obs_plan_embedding = obs_plan_embedding.unsqueeze(0)

    # Retriever를 이용하여 similarity 계산 (topk로 전체 결과 반환)
    similarity_results = retriever.search_in_embeds(
        key_embeds=key_embeddings,
        query_embeds=obs_plan_embedding,
        topk=len(B),  # 모든 항목에 대한 점수 반환
        return_scores=True
    )

    similarity_results = sort_scores(similarity_results)

    # 전체 A 요소의 개수 (나중에 match score 정규화를 위해 사용)
    total_elements = len(A)

    # similarity_results['scores']는 리스트의 리스트 형태로 되어 있음
    if similarity_results['scores']:
        similarity_scores = similarity_results['scores'][0]
    else:
        similarity_scores = [0] * len(B)

    # similarity_scores는 이미 float 값이므로 .item() 호출 없이 정규화 수행
    max_similarity_score = max(similarity_scores, default=0)
    similarity_scores = [score / max_similarity_score if max_similarity_score else 0 for score in similarity_scores]

    # A의 요소와 B의 각 value_list 간의 매칭 횟수를 계산
    match_counts = [sum(1 for element in A if element in value_list) for _, (value_list, _) in B.items()]

    match_counts_relative = []
    for i, values in enumerate(B.values()):
        # values[0]가 리스트라고 가정
        match_counts_relative.append((match_counts[i] / (len(values[0]) + 1e-9)) * np.log((len(values[0]) + 1e-9)))

    max_match_count = max(match_counts_relative, default=0)
    normalized_match_scores = [count / max_match_count if max_match_count else 0 for count in match_counts_relative]

    # 결과 딕셔너리에 정규화된 match score와 similarity score를 저장
    for idx, (key, _) in enumerate(B.items()):
        results[key] = [normalized_match_scores[idx], similarity_scores[idx]]

    return results



def top_k_obs(input_dict, k):
    # 각 키의 리스트 값의 합계를 계산
    sum_dict = {key: sum(values) for key, values in input_dict.items()}
    # 합계를 기준으로 내림차순 정렬 후 상위 k개 키 반환
    sorted_keys = sorted(sum_dict, key=sum_dict.get, reverse=True)
    return sorted_keys[:k]
