
import json
from time import time
import getpass

from agent import GPTagent
from system_prompts import default_system_prompt, system_plan_agent, system_action_agent, completed_plan_prompt, exception_plan_prompt, turn_count_plan_prompt
from memory.graph import ContrieverGraph
from memory.plot import plot_contriever_graph




def Logger(log_file):
    def log_func(msg):
        print(msg)
    return log_func

log_file = "goallm_v7"
model = "gpt-4o-2024-11-20"
#model = "gpt-4o-mini-2024-07-18"
api_key = getpass.getpass("Enter your OpenAI API Key: ")
n_prev, topk_episodic = 5, 2
max_steps, n_attempts = 20, 1  # 최대 턴 수를 20으로 설정 (예시)

curr_location = "Inside the Operations Conference Barracks"
observation = "A user and an NPC are talking inside the Operations Conference Barracks. They are sitting at the same table."
main_goal = "Build rapport with each other."
valid_actions = ["look", "sit", "stand", "attack", "hand shake"]

log = Logger(log_file)

# 에이전트 생성
agent = GPTagent(model=model, system_prompt=default_system_prompt, api_key=api_key)
agent_plan = GPTagent(model=model, system_prompt=system_plan_agent, api_key=api_key)
agent_action = GPTagent(model=model, system_prompt=system_action_agent, api_key=api_key)

# 초기 plan_agent 출력 예시 (최초 계획)
initial_plan0 = f'''{{
"main_goal": "서로 친밀감을 형성하라.",
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
    # 해당 step_number에 해당하는 항목에 "status": "completed"를 추가합니다.
    for step in plan_json.get("plan_steps", []):
        if step.get("step_number") == step_number:
            step["status"] = "completed"
            # reason에 완료 표시를 덧붙일 수 있음
            step["reason"] = step.get("reason", "") + " (completed)"
    return plan_json

def planning(condition, observations, observation, relevant_episodes, related_topics, previous_plan_json):
    prompt = f"""
1. Main goal: {main_goal}
2. History: {observations}
3. Current observation: {observation}
4. Relevant episodes: {relevant_episodes}
5. Related topics: {related_topics}
6. Previous plan: {previous_plan_json}
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

def choose_action(observations, observation_with_conversation, relevant_episodes, related_topics, current_plan, valid_actions):
    prompt = f"""
1. Main goal: {main_goal}
2. History: {observations}
3. Latest observation: {observation_with_conversation}
4. Relevant episodes: {relevant_episodes}
5. Related topics: {related_topics}
6. Current plan (Focus on not-completed step!): {current_plan}
7. Possible actions: {valid_actions}
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
    #log("Action Agent Response: " + action_output)
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
    # 환경 초기화
    action = "start"
    plan0 = initial_plan0  # 초기 plan0 문자열 (예: 자기소개 중심의 plan)
    log(f"Initial plan: {plan0}")
    count = 0
    exception_flag = False
    total_time = 0
    observations, history = [], []
    subgraph = []
    locations = set()
    locations.add(curr_location)

    # 이전 턴의 NPC 응답 저장 변수 (초기엔 빈 문자열)
    prev_npc = ""

    # ContrieverGraph 인스턴스 생성 (대화 내용 포함)
    graph = ContrieverGraph(model, system_prompt="You are a helpful assistant", api_key=api_key, device="cpu", debug=False)

    for turn in range(max_steps):
        turn_start = time()
        log(f"Turn: {turn+1}")
        count += 1

        # 1. 사용자 입력 받기
        user_input = input("User: ")
        if user_input=="quit": break

        # observation에 이전 턴의 NPC 응답이 있다면 포함시킴
        observation_with_conversation = ""
        if prev_npc:
            observation_with_conversation += "\nNPC: " + prev_npc
        observation_with_conversation += observation + "\nUser: " + user_input

        # 2. 아이템 처리
        observed_items, _ = agent.item_processing_scores(observation_with_conversation, plan0)
        items = {key.lower(): value for key, value in observed_items.items()}
        log("Crucial items: " + str(items))

        # 3. ContrieverGraph 업데이트 (대화 내용 포함, plan context 기반 episodic memory 업데이트)
        graph_update_start = time()
        subgraph, top_episodic = graph.update(observation_with_conversation, plan=plan0,
                                              prev_subgraph=subgraph,
                                              locations=list(locations),
                                              action=action, log=log,
                                              items1=items, topk_episodic=topk_episodic)
        graph_update_time = time() - graph_update_start
        log("Associated triplets: " + str(subgraph))
        log("Episodic memory: " + str(top_episodic))
        log(f"Time for graph update: {graph_update_time:.2f} sec")

        # 4. 행동 선택 (relevant_episodes, related_topics 포함)
        action_selection_start = time()
        npc_response, action, completed_step, exception_flag = choose_action(
            history, observation_with_conversation, top_episodic, subgraph, plan0, valid_actions)
        action_selection_time = time() - action_selection_start
        log("NPC: " + npc_response)
        log("Action: " + action)
        log("Completed step: " + str(completed_step))
        log(f"Time for action selection: {action_selection_time:.2f} sec")

        # 5. 기록 업데이트 및 plan 상태 업데이트 (완료된 step은 'completed' 상태로 표시)
        combined_entry = f"Observation: {observation_with_conversation}\nNPC: {npc_response}\nAction: {action}"
        observations.append(observation_with_conversation)
        history.append(combined_entry)
        previous_location = curr_location
        if completed_step != -1:
            log(f"Plan step {completed_step} completed. Marking as completed in the plan.")
            plan_current = json.loads(plan0)
            plan_current = mark_completed_step(plan_current, completed_step)
            plan0 = json.dumps(plan_current)

        # 6. 플래닝 호출 조건: 턴이 5 이상, exception flag 발생, 또는 plan_steps 모두 완료된 경우
        current_plan_json = json.loads(plan0)
        all_steps_completed = all(step.get("status") == "completed" for step in current_plan_json.get("plan_steps", []))
        if (count >= 5) or exception_flag or all_steps_completed:
            if count >= 5: condition = 'count 5'
            if exception_flag: condition = 'excepsion'
            if all_steps_completed: condition = 'finish plan'
            log(f"Plan Agent 재실행 (조건: {condition})")
            planning_start = time()
            plan_response = planning(condition, history, observation_with_conversation, top_episodic, subgraph, plan0)
            plan0 = plan_response  # plan0를 새로 갱신
            planning_time = time() - planning_start
            log(f"Time for planning: {planning_time:.2f} sec")
            exception_flag = False
            count = 0

        # 이번 턴의 NPC 응답을 다음 턴을 위해 저장
        prev_npc = npc_response

    # show final memory
    plot_contriever_graph(graph)
    for episode in graph.obs_episodic:
        print(episode)


if __name__ == "__main__":
    run()