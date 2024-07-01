import json
import openai


def gpt(args,
        prompt,
        max_tokens=128,
        temperature=0.3,
        stop=None,
        frequency_penalty=0,
        n=25):

    ## function to query LM ##
    # https://platform.openai.com/docs/api-reference/completions/create
    openai.api_key = args.openai_api_key
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=max_tokens,
                                        temperature=0.3,
                                        stop=stop,
                                        frequency_penalty=frequency_penalty,
                                        n=n)
    return response


def get_reordered_plan(args, task, commands_set):
    with open("data/pythonic_plans/train_complete_plan_set.json", 'r') as f:
        tmp = json.load(f)
        prompt_egs = {}
        for k, v in tmp.items():
            prompt_egs[k] = v

    f = open("scripts/example_selectors.txt", 'r', encoding='utf-8-sig')
    selectors = {}
    a_eg = ""
    ignore = ["assert", "#", "else"]
    for line in f:
        clean_line = line.strip()

        # Append the cleaned line to the list
        if clean_line != "--------------------------":
            if ":" in clean_line:
                commands = prompt_egs[clean_line[:-1].replace(" ", "_")].splitlines()
                a_eg += f"\n{commands[0]}"
                a_eg += f"\n    #Select lines of code from below and complete the function."
                a_eg += f"\n    #Multiple usages of a line and not using a line are both allowed."
                a_eg += f"\n    #Available lines of code:"
            else:
                if len(line) > 1:
                    a_eg += f"\n    #{clean_line}"
        else:
            a_eg += f"\n    #### YOUR CODE STARTS HERE"
            for command in commands[1:]:
                if not any(word in command for word in ignore):
                    if len(command) > 0:
                        a_eg += f"\n    {command.strip()}"
            a_eg += f"\n    #### YOUR CODE ENDS HERE"
            # print(a_eg)
            selectors[commands[0][:-1]] = a_eg
            a_eg = ""
    f.close()

    # prompt = f"from actions import turnright, turnleft, walkforward, walktowards <obj>, walk <obj>, run <obj>, grab <obj>, switchon <obj>, switchoff <obj>, open <obj>, close <obj>, lookat <obj>, sit <obj>, standup, find <obj>, turnto <obj>, drink <obj>, pointat <obj>, watch <obj>, putin <obj> <obj>, putback <obj> <obj>"
    # prompt = f"\n\nobjects = {['slippers', 'wallphone', 'nightstand', 'bedroom', 'kitchencounter', 'coffeemaker', 'dishwashingliquid', 'remotecontrol', 'plum', 'condimentshaker', 'lime', 'towelrack', 'bathroomcounter', 'window', 'closetdrawer', 'clothespile', 'garbagecan', 'fryingpan', 'sofa', 'hanger', 'deodorant', 'toothpaste', 'folder', 'hairproduct', 'rug', 'washingmachine', 'radio', 'tv', 'closet', 'kitchentable', 'stall', 'toothbrush', 'doorjamb', 'washingsponge', 'livingroom', 'whippedcream', 'cereal', 'chips', 'perfume', 'photoframe', 'sink', 'towel', 'wineglass', 'crackers', 'kitchencabinet', 'walllamp', 'bathroom', 'clothespants', 'peach', 'toaster', 'bananas', 'bed', 'toilet', 'bench', 'keyboard', 'ceiling', 'ceilinglamp', 'cutleryknife', 'clothesshirt', 'desk', 'barsoap', 'mousemat', 'kitchen', 'wallshelf', 'floor', 'oventray', 'bookshelf', 'microwave', 'cupcake', 'plate', 'stovefan', 'mouse', 'door', 'clock', 'breadslice', 'tablelamp', 'chocolatesyrup', 'orchid', 'salmon', 'apple', 'candybar', 'candle', 'bellpepper', 'mug', 'waterglass', 'cutleryfork', 'chair', 'bathroomcabinet', 'cpuscreen', 'wallpictureframe', 'pie', 'curtains', 'dishbowl', 'tvstand', 'creamybuns', 'coffeepot', 'faucet', 'wall', 'coffeetable', 'book', 'condimentbottle', 'bathtub', 'painkillers', 'pillow', 'cabinet', 'stove', 'computer', 'fridge', 'powersocket', 'paper', 'lightswitch', 'facecream', 'box', 'kitchencounterdrawer', 'cellphone']}"
    prompt = f"#if sit, need stand before walk"
    prompt+=f"\n\n"
    for k in ["def put_the_wine_glass_in_the_kitchen_cabinet()",
                            "def throw_away_the_lime()",
                            "def wash_mug()",]:
        prompt += f"{selectors[k]}"
        prompt += f"\n"
    prompt += f"def {task}():\n    #Select lines of code from below and complete the function."
    prompt += f"\n    #Contents above are examples."
    prompt += f"\n    #Follow the logics of the examples presented above. Try to make a valid and complete plan."
    prompt += f"\n    #Multiple usages of a line and not using a line are both allowed. It is also possible that some lines are not correct."
    prompt +=f"\n   Hint: Interacting with objects will not change their names."
    prompt += f"\n    #Available lines of code (You are only allowed to use these lines):"
    for line in commands_set:
        prompt += f"\n    #{line}"
    prompt += f"\n    #### YOUR CODE STARTS HERE"
    prompt += f"\n    #### YOUR CODE ENDS HERE ###"

    openai.api_key = args.openai_api_key
    print(f"Generating reordering plan for: {task}, with {args.gpt_reorder}")
    if "4" in args.gpt_reorder:
        response = openai.ChatCompletion.create(model=args.gpt_reorder,
                                                messages=[{"role": "user", "content": prompt}],
                                                max_tokens=300,
                                                temperature=0.7,
                                                frequency_penalty=0.1,
                                                n=args.num_plans_reorder)
    else:
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo',#args.gpt_reorder,
                                                messages=[{"role": "user", "content": prompt}],
                                                max_tokens=300,
                                                temperature=0.6,
                                                frequency_penalty=0.1,
                                                n=args.num_plans_reorder)

    plan = []
    for i in range(args.num_plans_reorder):
        plan.append(response["choices"][i]["message"]["content"].strip().split('\n'))
    return plan


def parse_step(line):
    if "(" not in line:
        return None, None
    ind = line.index("(")
    action = line[:ind]
    line = line[ind+1:-1]
    if "," not in line:
        obj = [line[1:-1]]
    else:
        temp = line.split(",")
        obj = [temp[0].strip()[1:-1], temp[1].strip()[1:-1]]

    return action, obj
