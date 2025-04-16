import os
import re
import ast
import json
import numpy as np

def clear_triplet(triplet):
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


def top_k_obs(input_dict, k):
    # 각 키의 리스트 값의 합계를 계산
    sum_dict = {key: sum(values) for key, values in input_dict.items()}
    # 합계를 기준으로 내림차순 정렬 후 상위 k개 키 반환
    sorted_keys = sorted(sum_dict, key=sum_dict.get, reverse=True)
    return sorted_keys[:k]
