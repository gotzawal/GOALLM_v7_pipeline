import json
import torch
import torch.nn.functional as F
from time import time
import getpass
from collections import OrderedDict

# (GPTagent, ContrieverGraph, Retriever, get_cached_embeddings, 그리고 system_prompt 등 필요한 모듈/상수들은 이미 정의되었다고 가정)

def Logger(log_file):
    def log_func(msg):
        print(msg)
    return log_func

log_file = "goallm_v7"
default_model = "gpt-4o-2024-11-20"
mini_model = "gpt-4o-mini-2024-07-18"
good_model = "chatgpt-4o-latest"

api_key = getpass.getpass("Enter your OpenAI API Key: ")
n_prev, topk_episodic = 5, 2
max_steps, n_attempts = 20, 1  # 최대 턴 수를 20으로 설정 (예시)

curr_location = "Inside the Operations Conference Barracks"
observation = "A user and an NPC are talking inside the Operations Conference Barracks. They are sitting at the same table."
valid_actions = ["look", "sit", "stand", "attack", "hand shake"]

log = Logger(log_file)

# -- 에이전트 생성 --
agent = GPTagent(model=default_model, system_prompt=default_system_prompt, api_key=api_key)
agent_plan = GPTagent(model=default_model, system_prompt=system_plan_agent, api_key=api_key)
agent_action = GPTagent(model=good_model, system_prompt=system_action_agent, api_key=api_key)

agent_status = GPTagent(model=default_model, system_prompt=system_status_agent, api_key=api_key)

# 에너지 및 신뢰도에 따른 보충 설명 프롬프트
energy_prompts = {
    "매우낮음": "현재 에너지가 매우 낮습니다. 휴식이나 재충전이 시급합니다.",
    "낮음": "현재 에너지가 낮습니다. 에너지 관리에 신경 쓸 필요가 있습니다.",
    "보통": "현재 에너지가 보통입니다. 무난한 상태를 유지하고 있습니다.",
    "높음": "현재 에너지가 높습니다. 임무에 적극적으로 임할 준비가 되어 있습니다.",
    "매우높음": "현재 에너지가 매우 높습니다. 극도의 집중력과 추진력이 발휘되고 있습니다."
}

trust_prompts = {
    "매우낮음": "사용자와의 신뢰도가 매우 낮습니다. 즉각적인 신뢰 회복 전략이 필요합니다.",
    "낮음": "사용자와의 신뢰도가 낮습니다. 신중하게 접근해 신뢰를 회복해야 합니다.",
    "보통": "사용자와의 신뢰도가 보통입니다. 기본 신뢰 형성은 되었지만, 보강이 필요합니다.",
    "높음": "사용자와의 신뢰도가 높습니다. 안정적인 신뢰 관계를 유지하고 있습니다.",
    "매우높음": "사용자와의 신뢰도가 매우 높습니다. 강한 신뢰 기반을 활용해 전략을 실행할 수 있습니다."
}

# 초기 plan_agent 출력 예시 (최초 계획)
initial_plan0 = f'''{{
"plan_steps": [
    {{
      "step_number": 1,
      "sub_goal": "사용자(병사)에게 반말로 인사하며, 간단한 자기소개와 배경을 나눈다.",
      "reason": "서로에 대해 알아가며 신뢰를 형성하고, 편안한 대화 분위기를 만든다.",
      "status": "not completed"
    }},
    {{
      "step_number": 2,
      "sub_goal": "본인(자신)의 간단한 자기소개를 진행한다.",
      "reason": "자신의 역할, 배경 및 특성을 명확히 하여 상대방과의 대화에 도움을 준다.",
      "status": "not completed"
    }}
],
"your_emotion": {{
      "your_current_emotion": "curious and friendly",
      "reason_behind_emotion": "새로운 동료와 처음 만나 서로에 대해 알아가는 과정에 기대와 호기심을 느낀다."
}},
"context_info": "현재 상황: 처음 만난 상태에서 전투 전략보다는 서로의 자기소개에 집중하여 친밀감을 형성하는 것이 중요하다.",
"reactive_plan": "대화가 원활하게 진행되지 않거나 계획에 없는 상황이 발생하면, 당황하거나, 머뭇거리거나 얼버무리며 exception flag를 호출한다."
}}'''

def mark_completed_step(plan_json, step_number):
    # 해당 step_number에 해당하는 항목에 "status": "completed"를 추가
    for step in plan_json.get("plan_steps", []):
        if step.get("step_number") == step_number:
            step["status"] = "completed"
            step["reason"] = step.get("reason", "") + " (completed)"
    return plan_json

# 최근 5턴 동안의 관련 지식을 중복 주제 없이 보관 (OrderedDict 사용)
recent_knowledge = OrderedDict()

# 지식 검색용 Retriever (Retriever 클래스가 정의되어 있다고 가정)
knowledge_retriever = Retriever(device='cpu')

#########################################################
# 1. 상태 관리 agent (get_status) 구현
#    planning 호출 시 누적 history를 기반으로 상태 평가
#########################################################
def get_status(history_context):
    """
    history_context: 누적된 history(문자열)를 받아서 agent_status를 통해
    JSON 형식의 상태 패널을 반환합니다.
    출력 예시:
      {
         "mental_energy": "보통",
         "user_trust": "높음",
         "current_task": "대화"
      }
    """
    prompt = system_status_agent + "\n" + history_context
    status_output, cost_status = agent_status.generate(prompt, jsn=True, t=0.5)
    try:
        status_json = json.loads(status_output)
    except Exception as e:
        status_json = {
            "mental_energy": "보통",
            "user_trust": "높음",
            "current_task": "대화"
        }
    log("Status: "+str(status_json))
    return status_json

#########################################################
# 2. 각 작업에 따른 추가 프롬프트 (작업 지침) 정의
#########################################################
status_prompts = {
    "정비": {
        "planning": "현재는 정비 모드입니다. 시스템 점검, 자원 정리 및 현재 상태를 재평가할 필요가 있습니다. 명확한 문제 진단과 안정적인 상태 유지를 위한 계획을 수립하세요.",
        "action": "정비 모드에서는 행동보다는 상태 점검 및 조정에 집중합니다. 침착하게 주변을 관찰하고 필요한 정리 작업을 우선시 하세요."
    },
    "대화": {
        "planning": "현재는 대화 모드입니다. 상호 신뢰를 형성하고, 친밀감을 높이기 위한 대화 전략을 구상하세요. 상대방의 감정을 고려한 부드러운 접근이 필요합니다.",
        "action": "대화 모드에서는 친절하고 부드러운 언어로 상대와 소통하며, 감정 표현에 신경 쓰세요."
    },
    "수행": {
        "planning": "현재는 수행 모드입니다. 본격적인 목표 달성을 위한 구체적인 실행 계획을 수립하세요. 상황 분석과 신속한 결정을 내리는 것이 중요합니다.",
        "action": "수행 모드에서는 신속하고 결단력 있게 행동하여 목표 달성을 위한 구체적인 실행 단계를 밟으세요."
    }
}

#########################################################
# 3. planning 함수 수정: 상태창 정보를 프롬프트에 포함
#    에너지와 신뢰도 보충 설명도 함께 추가
#########################################################
def planning(condition, observations, observation, relevant_episodes, related_topics, previous_plan_json, related_knowledge, status):
    """
    status: get_status()로부터 받은 상태 패널 딕셔너리
    """
    # 에너지와 신뢰도에 따른 보충 설명 생성
    supplementary_energy = energy_prompts.get(status["mental_energy"], "")
    supplementary_trust = trust_prompts.get(status["user_trust"], "")

    status_info = f"""상태창:
    - 정신적 에너지: {status['mental_energy']} ({supplementary_energy})
    - 사용자 신뢰도: {status['user_trust']} ({supplementary_trust})
    - 현재 작업: {status['current_task']}
추가 Planning 지침: {status_prompts[status['current_task']]['planning']}"""

    prompt = f"""
1. State: {status_info}
2. History: {observations}
3. Current observation: {observation}
4. Relevant episodes: {relevant_episodes}
5. Related topics: {related_topics}
6. Previous plan: {previous_plan_json}
7. Predefined Knowledge: {related_knowledge}
"""
    if condition == 'count 5':
        prompt += turn_count_plan_prompt
    elif condition == 'excepsion':
        prompt += exception_plan_prompt
    else:
        prompt += completed_plan_prompt

    plan_output, cost_plan = agent_plan.generate(prompt, jsn=True, t=0.6)
    log("Plan Agent Response: " + plan_output)
    return plan_output

#########################################################
# 4. choose_action 함수 수정: 상태창 정보를 프롬프트에 포함
#    (에너지와 신뢰도 보충 설명 추가)
#########################################################
def choose_action(observations, observation_with_conversation, relevant_episodes, related_topics, current_plan, valid_actions, related_knowledge, status):
    supplementary_energy = energy_prompts.get(status["mental_energy"], "")
    supplementary_trust = trust_prompts.get(status["user_trust"], "")

    status_info = f"""상태창:
    - 정신적 에너지: {status['mental_energy']} ({supplementary_energy})
    - 사용자 신뢰도: {status['user_trust']} ({supplementary_trust})
    - 현재 작업: {status['current_task']}
추가 행동 지침: {status_prompts[status['current_task']]['action']}"""

    prompt = f"""
1. State: {status_info}
2. History: {observations}
3. Latest observation: {observation_with_conversation}
4. Relevant episodes: {relevant_episodes}
5. Related topics: {related_topics}
6. Current plan (Focus on not-completed step!): {current_plan}
7. Predefined Knowledge: {related_knowledge}
8. Possible actions: {valid_actions}
Use context, reactive plan, and any additional information as needed.
Generate a JSON object exactly in the following format:
{{
  "action": "One selected action from the provided list.",
  "npc_response": "A concise dialogue line that is context-aware.",
  "completed_step": <number>,
  "exception_flag": <boolean>
}}
Do not write anything else.
"""
    t = 1
    action_output, cost_action = agent_action.generate(prompt, jsn=True, t=t)
    try:
        action_json = json.loads(action_output)
        npc_response = action_json["npc_response"]
        action = action_json["action"]
        completed_step = action_json["completed_step"]
        exception_flag = action_json["exception_flag"]
    except Exception as e:
        log("!!!INCORRECT ACTION CHOICE!!!")
        npc_response = "죄송합니다, 이해하지 못했어요."
        action = "look"
        completed_step = -1
        exception_flag = False
    return npc_response, action, completed_step, exception_flag


def run():
    action = "start"
    plan0 = initial_plan0
    log(f"Initial plan: {plan0}")
    count = 0
    exception_flag = False
    observations, history = [], []
    subgraph = []  # 이전 subgraph (prev_subgraph)
    locations = set()
    locations.add(curr_location)
    prev_npc = ""

    # 초기 기본 상태 (planning 호출 전까지는 기본 상태 사용)
    current_status = {
        "mental_energy": "보통",
        "user_trust": "보통",
        "current_task": "대화"
    }

    graph = ContrieverGraph(default_model, system_prompt="You are a helpful assistant", api_key=api_key, device="cpu", debug=False)

    for turn in range(max_steps):
        turn_start = time()
        log(f"Turn: {turn+1}")
        count += 1

        # 1. 사용자 입력 받기
        user_input = input("User: ")
        if user_input == "quit":
            break

        retrieve_start = time()

        # 이전 NPC 응답 포함 및 observation 구성
        observation_with_conversation = ""
        if prev_npc:
            observation_with_conversation += "\nNPC: " + prev_npc
        observation_with_conversation += observation + "\nUser: " + user_input

        # 2. 아이템 처리
        observed_items, _ = agent.item_processing_scores(observation_with_conversation, plan0)
        items = {key.lower(): value for key, value in observed_items.items()}
        log("Crucial items: " + str(items))

        # 3. Memory retrieve: 최근 트리플릿과 현재 observation 기반 검색
        retrieved_subgraph, top_episodic = graph.memory_retrieve(observation_with_conversation, plan0, subgraph, recent_n=5, topk_episodic=topk_episodic)
        log("Retrieved associated subgraph: " + str(retrieved_subgraph))
        log("Retrieved top episodic memory: " + str(top_episodic))

        # --- RAG: 사전에 정의한 지식을 이용해 관련 지식 업데이트 ---
        related_knowledge_items = filter_items_by_similarity(predefined_knowledge, observation_with_conversation, threshold=0.37, retriever=knowledge_retriever, max_n= 3)
        for subject, content, score in related_knowledge_items:
            recent_knowledge[subject] = content  # 중복된 주제는 덮어쓰기됨
            recent_knowledge.move_to_end(subject)
            log("Related knowledge: "+str(subject)+str(content)+" (score: "+str(score))
        while len(recent_knowledge) > 5:
            recent_knowledge.popitem(last=False)
        combined_knowledge_str = "; ".join([f"{subj}: {cont}" for subj, cont in recent_knowledge.items()])

        retrieve_time = time() - retrieve_start
        log(f"Time for retrieve: {retrieve_time:.2f} sec")

        # 4. 행동 선택: memory_retrieve 결과를 활용하여 choose_action 호출
        # relevant_episodes에는 top_episodic, related_topics에는 retrieved_subgraph 전달
        action_selection_start = time()
        npc_response, action, completed_step, exception_flag = choose_action(
            history,
            observation_with_conversation,
            top_episodic,
            retrieved_subgraph,
            plan0,
            valid_actions,
            related_knowledge=combined_knowledge_str,  # 필요시 관련 지식 문자열 사용
            status=current_status
        )
        action_selection_time = time() - action_selection_start
        log("NPC: " + npc_response)
        log("Action: " + action)
        log("Completed step: " + str(completed_step))
        log(f"Time for action selection: {action_selection_time:.2f} sec")

        # 5. 기록 업데이트
        combined_entry = f"Observation: {observation_with_conversation}\nNPC: {npc_response}\nAction: {action}"
        observations.append(observation_with_conversation)
        history.append(combined_entry)
        if len(history) > n_prev:
            history = history[-n_prev:]

        if completed_step != -1:
            log(f"Plan step {completed_step} completed. Marking as completed in the plan.")
            plan_current = json.loads(plan0)
            plan_current = mark_completed_step(plan_current, completed_step)
            plan0 = json.dumps(plan_current)

        # 6. 플래닝 조건 확인 및 재실행: 일정 턴 후 or exception 발생 시
        current_plan_json = json.loads(plan0)
        all_steps_completed = all(step.get("status") == "completed" for step in current_plan_json.get("plan_steps", []))
        if (count >= 5) or exception_flag or all_steps_completed:
            # 누적된 history 기반 상태 업데이트
            history_context = "\n".join(history)
            current_status = get_status(history_context)
            log(f"Updated status (from planning): {current_status}")

            if count >= 5:
                condition = 'count 5'
            elif exception_flag:
                condition = 'excepsion'
            elif all_steps_completed:
                condition = 'finish plan'
            log(f"Plan Agent 재실행 (조건: {condition})")
            planning_start = time()
            plan_response = planning(condition, history, observation_with_conversation, top_episodic, retrieved_subgraph, plan0, related_knowledge=combined_knowledge_str, status=current_status)
            plan0 = plan_response
            planning_time = time() - planning_start
            log(f"Time for planning: {planning_time:.2f} sec")
            exception_flag = False
            count = 0

        # 7. update_without_retrieve: memory_retrieve 이후 새 정보 업데이트
        graph.update_without_retrieve(observation_with_conversation, plan0, subgraph, list(locations), action, items, log)
        # 업데이트 후 prev_subgraph 갱신 (필요에 따라)
        subgraph = retrieved_subgraph

        # 이번 턴의 NPC 응답 저장
        prev_npc = npc_response

    # 최종 결과: ContrieverGraph 시각화 및 episodic memory 출력
    plot_contriever_graph(graph)
    for episode in graph.obs_episodic:
        print(episode)


if __name__ == "__main__":
    run()
