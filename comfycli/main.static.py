from cli_args import parse_cli_args
from execution import execute_prompt


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
        if checkpoint is not None:
            workflow = make_dict()
            node_ckpt = make_dict()
            dict_set(node_ckpt, "class_type", "CheckpointLoaderSimple")
            ckpt_inputs = make_dict()
            dict_set(ckpt_inputs, "ckpt_name", checkpoint)
            dict_set(node_ckpt, "inputs", ckpt_inputs)
            dict_set(workflow, "1", node_ckpt)
            content = json_dumps(workflow)
            result = execute_prompt(content, output_dir)
        else:
            print("Usage: comfycli workflow.json --output-dir ./output")
            print("   or: comfycli --checkpoint model.safetensors --prompt 'cat' --output ./out.png")
