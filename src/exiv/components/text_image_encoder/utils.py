import torch

def replace_escaped_parantheses(text):
    # runs before the main prompt parser
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text

def restore_escaped_parantheses(text):
    # runs after the main prompt parser
    # restores the change done by replace_escaped_parantheses
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text

def parse_parentheses(string):
    '''
    segments the prompt based on the parantheses
    input: 'A (very (fast) red) car'
    output: ['A ', '(very (fast) red)', ' car']
    '''
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result

def weighted_tokens(string, current_weight):
    '''
    recursively gets the weight of tokens
    input: "A (very) (cute:1.5) puppy"
    output: [('A ', 1.0), ('very', 1.1), ('cute', 1.5), (' puppy', 1.0)]
    '''
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += weighted_tokens(x, weight)
        else:
            out += [(x, current_weight)]
    return out

def bundled_embed(embed, prefix, suffix): #bundled embedding in lora format
    out_list = []
    for k in embed:
        if k.startswith(prefix) and k.endswith(suffix):
            out_list.append(embed[k])
    if len(out_list) == 0:
        return None

    return torch.cat(out_list, dim=0)