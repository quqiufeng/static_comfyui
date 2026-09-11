from cli_args import parse_cli_args
from execution import execute_prompt


def guard_main():
    main()


def make_workflow_node(class_type: str, inputs) -> dict:
    node = make_dict()
    dict_set(node, "class_type", class_type)
    dict_set(node, "inputs", inputs)
    return node


def build_prompt_workflow(checkpoint: str, prompt: str, output_path: str, output_dir: str,
                          width: int, height: int, steps: int, cfg: float,
                          seed: int, sampler: str, scheduler: str) -> str:
    # Determine output directory and filename prefix.
    if output_path is not None and str_length(output_path) > 0:
        out_dir = path_dirname(output_path)
        if str_length(out_dir) == 0:
            out_dir = "."
        parts = path_split(output_path)
        filename = parts[1]
        # Strip .png extension if present.
        if str_ends_with(filename, ".png"):
            filename = str_slice(filename, 0, str_length(filename) - 4)
        filename_prefix = filename
    else:
        out_dir = output_dir
        filename_prefix = "comfy_cli"

    # Default external CLIP encoders for SDXL.
    clip_l = "clip_l.safetensors"
    clip_g = "clip_g.safetensors"

    workflow = make_dict()

    ckpt_inputs = make_dict()
    dict_set(ckpt_inputs, "ckpt_name", checkpoint)
    dict_set(ckpt_inputs, "clip_l_name", clip_l)
    dict_set(ckpt_inputs, "clip_g_name", clip_g)
    dict_set(workflow, "1", make_workflow_node("CheckpointLoaderSimple", ckpt_inputs))

    sampler_inputs = make_dict()
    dict_set(sampler_inputs, "model", py_list("1", 0))
    dict_set(sampler_inputs, "prompt", prompt)
    dict_set(sampler_inputs, "negative_prompt", "")
    dict_set(sampler_inputs, "width", width)
    dict_set(sampler_inputs, "height", height)
    dict_set(sampler_inputs, "steps", steps)
    dict_set(sampler_inputs, "cfg", cfg)
    dict_set(sampler_inputs, "sampler_name", sampler)
    dict_set(sampler_inputs, "scheduler", scheduler)
    dict_set(sampler_inputs, "seed", seed)
    dict_set(sampler_inputs, "output_dir", out_dir)
    dict_set(sampler_inputs, "filename_prefix", filename_prefix)
    dict_set(workflow, "2", make_workflow_node("KSampler", sampler_inputs))

    save_inputs = make_dict()
    dict_set(save_inputs, "images", py_list("2", 0))
    dict_set(workflow, "3", make_workflow_node("SaveImage", save_inputs))

    return json_dumps(workflow)


def main():
    args = parse_cli_args()
    show_help: bool = dict_get(args, "show_help")
    if show_help:
        print_help()
        exit_program(0)
    output_dir = dict_get(args, "output_dir")
    if output_dir is None:
        output_dir = "./output"
    workflow_path = dict_get(args, "workflow")
    if workflow_path is not None and str_length(workflow_path) > 0:
        content = file_read_all(workflow_path)
        result = execute_prompt(content, output_dir)
    else:
        prompt = dict_get(args, "prompt")
        checkpoint = dict_get(args, "checkpoint")
        output_path = dict_get(args, "output")
        width = get_int(args, "width", 1024)
        height = get_int(args, "height", 1024)
        steps = get_int(args, "steps", 20)
        cfg = get_float(args, "cfg", 7.0)
        seed = get_int(args, "seed", 42)
        sampler = get_str(args, "sampler", "euler_a")
        scheduler = get_str(args, "scheduler", "discrete")
        if checkpoint is not None and prompt is not None:
            content = build_prompt_workflow(checkpoint, prompt, output_path, output_dir,
                                              width, height, steps, cfg, seed,
                                              sampler, scheduler)
            result = execute_prompt(content, output_dir)
        else:
            print("Usage: comfycli-bin workflow.json --output-dir ./output")
            print("   or: comfycli-bin --checkpoint model.safetensors --prompt 'cat' --output ./out.png")
            exit_program(1)
