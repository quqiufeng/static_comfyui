from nodes import call_node


def build_deps(prompt):
    node_ids = dict_keys(prompt)
    deps = make_dict()
    inputs_cache = make_dict()
    n = len(node_ids)
    i = 0
    while i < n:
        nid = node_ids[i]
        node = dict_get(prompt, nid)
        raw_inputs = dict_get(node, "inputs")
        resolved = make_dict()
        dep_list = py_list()
        input_keys = dict_keys(raw_inputs)
        k = 0
        m = len(input_keys)
        while k < m:
            key = input_keys[k]
            val = dict_get(raw_inputs, key)
            if is_link(val):
                src_id = val[0]
                src_idx = val[1]
                dict_set(resolved, key, val)  # keep original [src_id, src_idx] array
                dep_list = py_list_append(dep_list, src_id)
            else:
                dict_set(resolved, key, val)
            k = k + 1
        dict_set(deps, nid, dep_list)
        dict_set(inputs_cache, nid, resolved)
        i = i + 1
    return deps, inputs_cache


def resolve_all(inputs, node_outputs):
    resolved = make_dict()
    keys = dict_keys(inputs)
    k = 0
    n = len(keys)
    while k < n:
        key = keys[k]
        val = dict_get(inputs, key)
        if is_link(val):
            src_id = val[0]
            src_idx = val[1]
            src_outputs = dict_get(node_outputs, src_id)
            resolved_val = src_outputs[src_idx]
        else:
            resolved_val = val
        dict_set(resolved, key, resolved_val)
        k = k + 1
    return resolved


def execute_prompt(prompt_json: str, output_dir: str):
    prompt = parse_json(prompt_json)
    node_ids = dict_keys(prompt)
    deps, inputs_cache = build_deps(prompt)
    node_outputs = make_dict()
    executed = make_dict()
    n = len(node_ids)
    remaining = n
    while remaining > 0:
        progress = 0
        i = 0
        while i < n:
            nid = node_ids[i]
            if dict_get(executed, nid) is None:
                ready = 1
                dep_list = dict_get(deps, nid)
                m = len(dep_list)
                j = 0
                while j < m:
                    dep_id = dep_list[j]
                    if dict_get(executed, dep_id) is None:
                        ready = 0
                    j = j + 1
                if ready == 1:
                    node = dict_get(prompt, nid)
                    class_type = dict_get(node, "class_type")
                    inputs = dict_get(inputs_cache, nid)
                    resolved = resolve_all(inputs, node_outputs)
                    print(class_type)
                    outputs = call_node(class_type, resolved)
                    dict_set(node_outputs, nid, outputs)
                    dict_set(executed, nid, 1)
                    remaining = remaining - 1
                    progress = 1
            i = i + 1
        if progress == 0:
            break
    return node_outputs


def main():
    pass
