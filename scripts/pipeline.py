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
import re
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

from utils_pipeline import *
from tree import Node, CommandIssuer
from utils_aug_env import get_obj_ids_for_adding_states, add_additional_obj_states


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

    # set number of plans generated for each task
    num_of_plans = args.num_plans_original

    gen_plan = {}
    if not args.load_generated_plans:
        f = open(f"{args.progprompt_path}/data/generate/{args.test_set}_pipeline.json", 'w')
        for task in test_tasks:
            print(f"Generating plan for: {task}\n")
            prompt_task = "def {fxn}():".format(fxn = '_'.join(task.split(' ')))
            curr_prompt = f"{prompt}\n\n{prompt_task}\n\t"
            response = gpt(args,
                           curr_prompt,
                           temperature=0.1,
                           max_tokens=600,
                           stop=["def"],
                           frequency_penalty=0.15,
                           n=num_of_plans)
            gen_plan[task] = [None] * num_of_plans
            for i in range(num_of_plans):
                gen_plan[task][i] = response["choices"][i]["message"]["content"].strip().split('\n')
        json.dump(gen_plan, f, indent=4)
        f.close()

    else:
        with open(f"{args.progprompt_path}/data/generate/{args.test_set}_pipeline.json", 'r') as f:
            gen_plan = json.load(f)

    commands_dict = {}
    ignore = ["assert", "----", "#", "def"]
    for task in test_tasks:
        commands = set()
        for plan in gen_plan[task]:
            for line in plan:
                # Strip newline characters from the end of each line
                clean_line = line.strip()
                # Append the cleaned line to the list
                if not any(word in clean_line for word in ignore):
                    if "else" in clean_line:
                        idx = clean_line.index(":")
                        clean_line = clean_line[idx+1:].strip()
                    commands.add(clean_line)
        commands_dict[task] = commands

    if not args.load_reordered_plans:
        plans = {}
        for task in test_tasks:
            if "4" in args.gpt_reorder:
                time.sleep(15)
            p = get_reordered_plan(args, task, commands_dict[task])
            plans[task] = p

        with open(f'{args.progprompt_path}/data/generate/{args.test_set}_reordered.json', 'w') as file:
            json.dump(plans, file, indent=4)

    else:
        with open(f'{args.progprompt_path}/data/generate/{args.test_set}_reordered.json', 'r') as file:
            plans = json.load(file)

    tree_list = []
    for i in range(len(test_tasks)):
        root = Node(None, None, True)
        tree_list.append(root)

    # best_plan = {}
    for task, i in zip(test_tasks, range(len(test_tasks))):

        actions=['turnright', 'turnleft', 'walkforward', 'walktowards', 'walk', 'run', 'grab', 'switchon', 'switchoff', 'open', 'close', 'lookat', 'sit', 'standup', 'find', 'turnto', 'drink', 'pointat', 'watch', 'putin', 'putback']

        objs=['slippers', 'wallphone', 'nightstand', 'bedroom', 'kitchencounter', 'coffeemaker', 'dishwashingliquid', 'remotecontrol', 'plum', 'condimentshaker', 'lime', 'towelrack', 'bathroomcounter', 'window', 'closetdrawer', 'clothespile', 'garbagecan', 'fryingpan', 'sofa', 'hanger', 'deodorant', 'toothpaste', 'folder', 'hairproduct', 'rug', 'washingmachine', 'radio', 'tv', 'closet', 'kitchentable', 'stall', 'toothbrush', 'doorjamb', 'washingsponge', 'livingroom', 'whippedcream', 'cereal', 'chips', 'perfume', 'photoframe', 'sink', 'towel', 'wineglass', 'crackers', 'kitchencabinet', 'walllamp', 'bathroom', 'clothespants', 'peach', 'toaster', 'bananas', 'bed', 'toilet', 'bench', 'keyboard', 'ceiling', 'ceilinglamp', 'cutleryknife', 'clothesshirt', 'desk', 'barsoap', 'mousemat', 'kitchen', 'wallshelf', 'floor', 'oventray', 'bookshelf', 'microwave', 'cupcake', 'plate', 'stovefan', 'mouse', 'door', 'clock', 'breadslice', 'tablelamp', 'chocolatesyrup', 'orchid', 'salmon', 'apple', 'candybar', 'candle', 'bellpepper', 'mug', 'waterglass', 'cutleryfork', 'chair', 'bathroomcabinet', 'cpuscreen', 'wallpictureframe', 'pie', 'curtains', 'dishbowl', 'tvstand', 'creamybuns', 'coffeepot', 'faucet', 'wall', 'coffeetable', 'book', 'condimentbottle', 'bathtub', 'painkillers', 'pillow', 'cabinet', 'stove', 'computer', 'fridge', 'powersocket', 'paper', 'lightswitch', 'facecream', 'box', 'kitchencounterdrawer', 'cellphone']

        for plan in plans[task]:
            pointer = tree_list[i]
            for step in plan:
                line = step.strip()
                if line == "" or "assert" in line or "else" in line \
                        or "def" in line or "#" in line:
                    continue
                act, obj = parse_step(line)
                if act is None:
                    continue
                # if act not in actions:
                #     continue
                # if obj[0] not in objs:
                #     continue
                if len(obj) > 1 and obj[1] not in objs:
                    continue
                pointer = pointer.add_child_by_ao(line)
        # pointer = tree_list[i]
        # res = []
        # while len(pointer.children) > 0:
        #     pointer = pointer.get_child_highest_vote()
        #     if pointer is None:
        #         break
        #     res.append(pointer.action)
        # best_plan[task] = res

    final_states, initial_states, exec_per_task = run_exec(args,
                                                            comm,
                                                            test_tasks,
                                                            tree_list,
                                                            log_file)

    with open('data/final_states.json', 'r') as file:
        # Load its content and turn it into a Python dictionary
        data = json.load(file)

    success = []
    psr = []

    for fstate, istate, task in zip(final_states, initial_states, test_tasks):
        obj_ids_i = dict([(node['id'], node['class_name']) for node in fstate['nodes']])
        relations_in_i = set([obj_ids_i[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids_i[n['to_id']] for n in fstate["edges"]])
        obj_states_in_i = set([node['class_name'] + ' ' + st for node in fstate['nodes'] for st in node['states']])

        obj_ids = dict([(node['id'], node['class_name']) for node in istate['nodes']])
        relations = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in istate["edges"]])
        obj_states = set([node['class_name'] + ' ' + st for node in istate['nodes'] for st in node['states']])

        # print(set(data[task][0]) - (relations_in_i-relations), set(data[task][1]) - (obj_states_in_i-obj_states))
        # recrod[task]=(list((relations_in_i-relations)), list(obj_states_in_i-obj_states))
        length = len(set(data[task][0]) - (relations_in_i-relations)) + len(set(data[task][1]) - (obj_states_in_i-obj_states))
        if length == 0:
            success.append(1)
        else:
            success.append(0)

        psr.append(1-length / (len(set(data[task][0])) + len(set(data[task][1]))))

    return final_states, initial_states, exec_per_task, test_tasks, success, psr


def run_exec(args, comm, test_tasks, tree_list, log_file):
    final_states = []
    initial_states = []
    exec_per_task = []

    for task, i in zip(test_tasks, range(len(test_tasks))):
        command_issuer = CommandIssuer(tree_list[i])
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
        action = ""
        while action is not None:
            if args.correction:
                action = command_issuer.get_next_step()
            else:
                action = command_issuer.get_next_highest_step()
            if action is None:
                continue
            print(action)
            # fixes needed for not getting stuck
            # if step > 10:
            #     break
            if "grab('wallphone')" in action:
                continue

            # since above steps are not for env, following line go through the env
            total_steps += 1

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
                    # print('no?',action)
                    step+1; log_file.write("obj not in hand\n"); 
                    print(1, "obj not in hand\n")
                    if args.correction:
                        command_issuer.execution_result(False)
                    continue
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
                        # print("no?",action)
                        step+1; log_file.write("obj not found\n"); 
                        print(2, "obj not found\n")
                        if args.correction:
                            command_issuer.execution_result(False)
                        continue
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
                    # print('no?',action)
                    step+1; log_file.write("obj not in hand\n"); 
                    print(3, "obj not in hand\n")
                    if args.correction:
                        command_issuer.execution_result(False)
                    continue
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
                        # print('no?', action)
                        step+1; log_file.write("obj not in hand\n"); 
                        print(4, "obj not in hand\n")
                        if args.correction:
                            command_issuer.execution_result(False)
                        continue
                #debug
                # print(obj_id1)
                found_id = min(obj_id1)
                script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], found_id)
            elif len(action)==1: # 0 object action
                script_instruction = '<char0> [{}]'.format(action[0])
            else:
                log_file.write("bad action\n"); 
                print(5, "bad action\n")
                if args.correction:
                    command_issuer.execution_result(False)
                continue

            ## execute next action in both envs: visual and graph
            log_file.write(f"{script_instruction}\n")
            _, m = comm.render_script([script_instruction], recording=True, find_solution=True)
            script = script_instruction[7:]
            # #debug
            # print(script)
            try:
                script = parse_script_line(script, 0)
            except:
                step+=1; 
                print(6, "error")
                if args.correction:
                    command_issuer.execution_result(False)
                continue
            print(script)
            success, final_state, _ = executor.execute(Script([script]))
            print(success)
            if args.correction:
                command_issuer.execution_result(success)

            if not success:
                print(executor.info.get_error_string(), "error")
                log_file.write(f"act_success: {success}, message: {executor.info.get_error_string()}\n")
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

    parser.add_argument("--gpt-reorder", type=str, default="gpt-3.5-turbo",
                        choices=['gpt-3.5-turbo', 'gpt-4'])
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

    parser.add_argument("--load-generated-plans", type=bool, default=False)

    parser.add_argument("--load-reordered-plans", type=bool, default=False)

    parser.add_argument("--num-plans-original", type=int, default=25)

    parser.add_argument("--num-plans-reorder", type=int, default=20)
    parser.add_argument("--correction", type=bool, default=False)

    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    if not osp.isdir(f"{args.progprompt_path}/results/new/"):
        os.makedirs(f"{args.progprompt_path}/results/new/")

    final_states, initial_states, exec_per_task, tasks, success , gcr= execute_plan(args)
    print(tasks)
    print(success)
    print("Exec:", sum(exec_per_task) / len(exec_per_task))
    print(exec_per_task)
    print("gcr", sum(gcr)/len(gcr))
    print(gcr)
