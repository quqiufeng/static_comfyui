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


def resolve_all(inputs, node_outputs, default_output_dir: str):
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
    # 节点未显式指定 output_dir 时，用 CLI --output-dir 作为默认
    if dict_get(resolved, "output_dir") is None:
        dict_set(resolved, "output_dir", default_output_dir)
    return resolved


def validate_prompt(prompt) -> int:
    # 对照 ComfyUI validate_inputs：节点类型存在 + 链接指向合法节点/输出
    node_ids = dict_keys(prompt)
    i = 0
    ni = len(node_ids)
    while i < ni:
        nid = node_ids[i]
        node = dict_get(prompt, nid)
        class_type = dict_get(node, "class_type")
        if not node_exists(class_type):
            print("validate: unknown node type '" + class_type + "' at node " + nid)
            return 1
        raw_inputs = dict_get(node, "inputs")
        input_keys = dict_keys(raw_inputs)
        k = 0
        nk = len(input_keys)
        while k < nk:
            key = input_keys[k]
            val = dict_get(raw_inputs, key)
            if is_link(val):
                src_id = val[0]
                src_idx = val[1]
                src_node = dict_get(prompt, src_id)
                if src_node is None:
                    print("validate: node " + nid + " input '" + key + "' links to missing node " + src_id)
                    return 2
                src_class = dict_get(src_node, "class_type")
                rc = node_return_count(src_class)
                if src_idx < 0 or src_idx >= rc:
                    print("validate: node " + nid + " input '" + key + "' links to invalid output " + string_of_int(src_idx) + " of " + src_class)
                    return 3
            k = k + 1
        i = i + 1
    return 0


def execute_prompt(prompt_json: str, output_dir: str):
    prompt = parse_json(prompt_json)
    vrc = validate_prompt(prompt)
    if vrc != 0:
        print("Prompt validation failed, rc=" + string_of_int(vrc))
        return make_dict()
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
                    resolved = resolve_all(inputs, node_outputs, output_dir)
                    outputs = call_node(class_type, resolved)
                    dict_set(node_outputs, nid, outputs)
                    dict_set(executed, nid, 1)
                    remaining = remaining - 1
                    progress = 1
            i = i + 1
        if progress == 0:
            break
    if remaining > 0:
        print("Prompt has a cycle or missing dependency: " + string_of_int(remaining) + " node(s) not executed")
    return node_outputs


def main():
    pass
