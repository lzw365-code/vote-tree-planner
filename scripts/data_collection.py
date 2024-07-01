import sys
sys.path.append("virtualhome/simulation")
sys.path.append("virtualhome/simulation/unity_simulator")
sys.path.append("virtualhome/demo")
sys.path.append("virtualhome")

import argparse
import os
import os.path as osp
import random
import pickle

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.demo.utils_demo import *
import numpy as np
from virtualhome.simulation.evolving_graph import utils
from virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome.simulation.evolving_graph.environment import EnvironmentGraph

import openai
import json
import time

from utils_execute import *


def execute_plan(args):
    comm = UnityCommunication(file_name=args.unity_filename, x_display=args.display)

    # prompt example environment is set to env_id 0
    comm.reset(0)
    _, env_graph = comm.environment_graph()
    obj = list(set([node['class_name'] for node in env_graph["nodes"]]))
    prompt = f"from actions import turnright, turnleft, walkforward, walktowards <obj>, walk <obj>, run <obj>, grab <obj>, switchon <obj>, switchoff <obj>, open <obj>, close <obj>, lookat <obj>, sit <obj>, standup, find <obj>, turnto <obj>, drink <obj>, pointat <obj>, watch <obj>, putin <obj> <obj>, putback <obj> <obj>"
    prompt += f"\n\nobjects = {obj}"
    with open(f"{args.progprompt_path}/data/pythonic_plans/train_complete_plan_set.json", 'r') as f:
        tmp = json.load(f)
        prompt_egs = {}
        for k, v in tmp.items():
            prompt_egs[k] = v
    
    if args.prompt_task_examples == "default":
        default_examples = ["put_the_wine_glass_in_the_kitchen_cabinet",
                            "throw_away_the_lime",
                            "wash_mug",
                            "refrigerate_the_salmon",
                            "bring_me_some_fruit",
                            "wash_clothes",
                            "put_apple_in_fridge"]
        for i in range(args.prompt_num_examples):
            prompt += "\n\n" + prompt_egs[default_examples[i]]

    # evaluate in given unseen env
    if args.env_id != 0:
        comm.reset(args.env_id)
        _, graph = comm.environment_graph()
        obj = list(set([node['class_name'] for node in graph["nodes"]]))
        prompt += f"\n\n\nobjects = {obj}"

        # evaluation tasks in given unseen env
        test_tasks = []
        with open(f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])
        # log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

    # setup logging
    log_filename = f"{args.expt_name}_{args.prompt_task_examples}_{args.prompt_num_examples}examples"
    if args.prompt_task_examples_ablation != "none":
        log_filename += f"_{args.prompt_task_examples_ablation}"
    log_filename += f"_{args.test_set}"
    log_file = open(f"{args.progprompt_path}/results/{log_filename}_logs.txt", 'w')
    log_file.write(f"\n----PROMPT for planning----\n{prompt}\n")
    
    # evaluate in seen env
    if args.env_id == 0:
        test_tasks = []
        for file in os.listdir(f"{args.progprompt_path}/data/{args.test_set}"):
            with open(f"{args.progprompt_path}/data/{args.test_set}/{file}", 'r') as f:
                for line in f.readlines():
                    test_tasks.append(list(json.loads(line).keys())[0])

        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

    gen_plan = {}
    test_tasks = []
    with open(f"{args.progprompt_path}/data/generate/{args.test_set}_plan.txt", 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(":"):
                test_tasks.append(line[0:-1])
                gen_plan[line[0:-1]] = []
            elif "(" in line and line[0]!="#":
                gen_plan[test_tasks[-1]].append(line)
    # print(test_tasks)
    # print(gen_plan)

    final_states, initial_states, exec_per_task = run_exec(args,
                                                            comm,
                                                            test_tasks,
                                                            gen_plan,
                                                            log_file)
    recrod={}
    with open('exp.json', 'r') as file:
    # Load its content and turn it into a Python dictionary
        data = json.load(file)
    for fstate,istate,task in zip(final_states,initial_states,test_tasks):
        obj_ids_i = dict([(node['id'], node['class_name']) for node in fstate['nodes']])
        relations_in_i = set([obj_ids_i[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids_i[n['to_id']] for n in fstate["edges"]])
        obj_states_in_i = set([node['class_name'] + ' ' + st for node in fstate['nodes'] for st in node['states']])

        obj_ids = dict([(node['id'], node['class_name']) for node in istate['nodes']])
        relations = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in istate["edges"]])
        obj_states = set([node['class_name'] + ' ' + st for node in istate['nodes'] for st in node['states']])

        print(set(data[task][0])-(relations_in_i-relations), set(data[task][1])-((obj_states_in_i-obj_states)))
        # recrod[task]=(list((relations_in_i-relations)), list(obj_states_in_i-obj_states))

    # with open("exp.json", "w") as f:
    #     json.dump(recrod,f)

    return final_states, initial_states, exec_per_task


def run_exec(args, comm, test_tasks, gen_plan, log_file):
    final_states = []
    initial_states = []
    exec_per_task = []

    for task in test_tasks:
        ## initialize and set up enviroenment: visual + graph environment ##
        comm.reset(args.env_id)
        comm.add_character('Chars/Male2', initial_room='kitchen')

        _, graph = comm.environment_graph()
        _, cc = comm.camera_count()
        initial_states.append(graph)

        env_graph = EnvironmentGraph(graph)
        name_equivalence = utils.load_name_equivalence()
        executor = ScriptExecutor(env_graph, name_equivalence)

        ## get agent's initial state ##
        agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
        agent_in_roomid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "INSIDE"][0]
        agent_in_room = [n['class_name'] for n in graph["nodes"] if n['id'] == agent_in_roomid][0]
        agent_has_objid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and "HOLD" in n["relation_type"]]
        agent_has_obj = [n['class_name'] for n in graph["nodes"] if n['id'] in agent_has_objid]
        # some actions might not execute in the visual simulation, but they will in evolving graphs
        images = []
        max_fails = 10; num_fails = 0
        _, im = comm.camera_image([cc-5], image_width=300, image_height=300)
        images.append(im[0])
        # s, obj = comm.get_visible_objects(cc-6)
        obj_ids_for_adding_states = get_obj_ids_for_adding_states(graph)
        nodes_with_additional_states = {}

        partial_graph = utils.get_visible_nodes(graph, agent_id=agent)

        ###
        # get the state of character
        #
        ###
        obj_ids_close = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and  n["relation_type"]=="CLOSE"]
        obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
        obj_ids = dict([(node['id'], node['class_name']) for node in graph['nodes'] if node["id"] in obj_ids_close and node['class_name'] in obj])
        relations = list(set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in graph["edges"] if n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"] not in ["CLOSE","FACING", "INSIDE"]]))    
        obj_states = [(node['class_name'], node['states']) for node in graph['nodes'] if node['class_name'] in obj]

        ## parse plan into subgoals ##
        log_file.write(f"\n--Executing task: {task}--\n")
        # log_file.write(f"Plan:  {plan}\n\n")
        print(f"Executing: {task}\n")

        ## begin execution ##
        executable_steps = 0; total_steps = 0
        step = 1; act = ""

        # print(task)
        # print(gen_plan)
        for action in gen_plan[task]:
            # fixes needed for not getting stuck
            # if step > 10:
            #     break
            if "grab('wallphone')" in action:
                continue
                
            # since above steps are not for env, following line go through the env
            total_steps+=1
            
            ## parse next action
            action = action.split(')')[0]
            action = re.findall(r"\b[a-z]+", action)
            found_id = None

            if len(action)==3 and "put" in action[0]: # 2 objs action
                obj_id1 = [node['id'] for node in partial_graph['nodes'] if node['class_name'] == action[1] and node['id'] in agent_has_objid]
                obj_id2 = [node['id'] for node in partial_graph['nodes'] if node['class_name'] == action[2]]
                if len(obj_id1)==0:
                    # obj_id1 = obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    # if len(obj_id1)==0:
                    print('no?')
                    step+1; log_file.write("obj not in hand\n"); continue
                if len(obj_id1)==1:
                    # #debugs
                    # print(id1)
                    id1 = obj_id1[0]
                else:
                    id1 = min(obj_id1)
                    # #debugs
                    # print(id1)
                
                if len(obj_id2)==0:
                    obj_id2 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[2]]
                    if len(obj_id2)==0:
                        print("no?")
                        step+1; log_file.write("obj not found\n"); continue
                if len(obj_id2)==1:
                    id2 = obj_id2[0]
                elif found_id in obj_id2:
                    id2 = found_id
                else:
                    id2 = min(obj_id2)
                script_instruction = '<char0> [{}] <{}> ({}) <{}> ({})'.format(action[0], action[1], id1, action[2], id2)
            elif len(action)==2 and action[0] not in ["find", "walk"]: # 1 obj action
                # print(partial_graph)
                obj_id1 = [node['id'] for node in partial_graph['nodes'] if node['class_name'] == action[1]]
                if len(obj_id1)==0:
                    # obj_id1 = obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    # if len(obj_id1)==0:
                    print('no?')
                    step+1; log_file.write("obj not in hand\n"); continue
                if len(obj_id1)==1:
                    id1 = obj_id1[0]
                elif found_id in obj_id1:
                    id1 = found_id
                # elif len(obj_id1)==0:
                #     print("no?")
                #     step+1; log_file.write("obj not found\n"); continue
                else:
                    id1 = min(obj_id1)
                script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], id1)
            elif len(action)==2: # walk or find action
                # ##debug
                # print("check")
                obj_id1 = [node['id'] for node in partial_graph['nodes'] if node['class_name'] == action[1]]
                # #debug
                # nd = [node for node in partial_graph['nodes'] if node['class_name'] == action[1]]
                # print(nd)
                if len(obj_id1)==0:
                    obj_id1 = obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    print(obj_id1)
                    if len(obj_id1)==0:
                        print('no?')
                        step+1; log_file.write("obj not in hand\n"); continue
                #debug
                # print(obj_id1)
                found_id = min(obj_id1)
                script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], found_id)
            elif len(action)==1: # 0 object action
                script_instruction = '<char0> [{}]'.format(action[0])
            else:
                log_file.write("bad action\n"); continue
            
            ## execute next action in both envs: visual and graph
            log_file.write(f"{script_instruction}\n")
            _, m = comm.render_script([script_instruction], recording=True, find_solution=True)
            script = script_instruction[7:]
            # #debug
            # print(script)
            try:
                script = parse_script_line(script, 0)
            except:
                step+=1; continue
            print(script)
            success, final_state, _ = executor.execute(Script([script])) 
            # time.sleep(3.0)
            
            if not success:
                print(executor.info.get_error_string())
                log_file.write(f"act_success: {success}, message: {executor.info.get_error_string()}\n")
                print("error")
            else:
                # count execution if action executes succesfully in graph env
                executable_steps+=1
                graph = final_state.to_dict()
                env_graph = EnvironmentGraph(graph)
                agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
                partial_graph = utils.get_visible_nodes(final_state.to_dict(), agent_id=agent)
                name_equivalence = utils.load_name_equivalence()
                executor = ScriptExecutor(env_graph, name_equivalence)
                script_instruction = ' '.join(re.findall(r"\b[a-z]+", script_instruction)[1:])
                step+=1

                # get new state info
                agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
                agent_in_roomid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "INSIDE"][0]
                agent_in_room = [n['class_name'] for n in graph["nodes"] if n['id'] == agent_in_roomid][0]
                agent_has_objid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and "HOLD" in n["relation_type"]]
                agent_has_obj = [n['class_name'] for n in graph["nodes"] if n['id'] in agent_has_objid]

                # Here you can get an observation, for instance 
                if 'grab' in script_instruction or 'open' in script_instruction or 'close' in script_instruction:
                    s, im = comm.camera_image([cc-5], image_width=300, image_height=300)
                else:
                    s, im = comm.camera_image([cc-6], image_width=300, image_height=300)
                images.append(im[0])

                obj_ids_close = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and  n["relation_type"]=="CLOSE"]
                obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
                obj_ids = dict([(node['id'], node['class_name']) for node in partial_graph['nodes'] if node["id"] in obj_ids_close and node['class_name']!=agent_in_room])
                nodes_with_additional_states = add_additional_obj_states(partial_graph, obj_ids_for_adding_states, nodes_with_additional_states)
                
        # augment state with additional state info
        if isinstance(final_state, dict):
            if "nodes" in final_state.keys():
                for idx in range(len(final_state["nodes"])):
                    if final_state["nodes"][idx]['id'] in nodes_with_additional_states.keys():
                        final_state["nodes"][idx] = nodes_with_additional_states[final_state["nodes"][idx]['id']]
        else:       
            final_state = final_state.to_dict()
            for idx in range(len(final_state["nodes"])):
                if final_state["nodes"][idx]['id'] in nodes_with_additional_states.keys():
                    final_state["nodes"][idx] = nodes_with_additional_states[final_state["nodes"][idx]['id']]

        # get final state for eval
        final_states.append(final_state)
        exec_per_task.append(executable_steps/total_steps)
    return final_states, initial_states, exec_per_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--progprompt-path", type=str,
                        default="/Users/lzw365/Documents/MyData/CSE_repo/progprompt-vh/")
    parser.add_argument("--expt-name", type=str, default="exp")

    parser.add_argument("--openai-api-key", type=str,
                        default="")    # put OpenAI API key here
    parser.add_argument("--unity-filename", type=str, 
                        default="/Users/lzw365/Documents/MyData/CSE_repo/progprompt-vh/virtualhome/virtualhome/simulation/unity_simulator/macos_exec.v2.3.0.app")
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument("--display", type=str, default="0")

    parser.add_argument("--gpt-version", type=str, default="gpt-3.5-turbo-instruct",
                        choices=['gpt-3.5-turbo-instruct', 'davinci-002', 'babbage-002'])
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--test-set", type=str, default="test_unseen", 
                        choices=['test_unseen', 'test_seen', 'test_unseen_ambiguous', 'env1', 'env2'])

    parser.add_argument("--prompt-task-examples", type=str, default="default",
                        choices=['default', 'random'])
    # for random task examples, choose seed
    parser.add_argument("--seed", type=int, default=0)

    ## NOTE: davinci or older GPT3 versions have a lower token length limit
    ## check token length limit for models to set prompt size: 
    ## https://platform.openai.com/docs/models
    parser.add_argument("--prompt-num-examples", type=int, default=3, 
                         choices=range(1, 7))
    parser.add_argument("--prompt-task-examples-ablation", type=str, default="none", 
                         choices=['none', 'no_comments', "no_feedback", "no_comments_feedback"])

    parser.add_argument("--load-generated-plans", type=bool, default=True)

    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    if not osp.isdir(f"{args.progprompt_path}/results/new/"):
        os.makedirs(f"{args.progprompt_path}/results/new/")

    final_states, initial_states, exec_per_task = execute_plan(args)
