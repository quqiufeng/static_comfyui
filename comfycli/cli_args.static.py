@dataclass
class CliArgs:
    checkpoint: str
    prompt: str
    output: str
    output_dir: str
    workflow: str
    show_help: bool
    cpu: bool
    cuda_device: str
    highvram: bool
    lowvram: bool


def parse_cli_args() -> dict:
    args_list: list[str] = argv()
    argc: int = py_list_length(args_list)

    checkpoint: str = ""
    prompt: str = ""
    output: str = ""
    output_dir: str = ""
    workflow: str = ""
    show_help: bool = False
    cpu: bool = False
    cuda_device: str = "0"
    highvram: bool = False
    lowvram: bool = False

    i: int = 1
    while i < argc:
        arg: str = py_list_ref(args_list, i)
        if arg == "--help" or arg == "-h":
            show_help = True
            i = i + 1
            continue
        elif arg == "--checkpoint" or arg == "--ckpt":
            i = i + 1
            if i < argc:
                checkpoint = py_list_ref(args_list, i)
        elif arg == "--prompt" or arg == "-p":
            i = i + 1
            if i < argc:
                prompt = py_list_ref(args_list, i)
        elif arg == "--output" or arg == "-o":
            i = i + 1
            if i < argc:
                output = py_list_ref(args_list, i)
        elif arg == "--output-dir" or arg == "--output_directory":
            i = i + 1
            if i < argc:
                output_dir = py_list_ref(args_list, i)
        elif arg == "--cpu":
            cpu = True
        elif arg == "--cuda-device" or arg == "--cuda_device":
            i = i + 1
            if i < argc:
                cuda_device = py_list_ref(args_list, i)
        elif arg == "--highvram" or arg == "--gpu-only":
            highvram = True
        elif arg == "--lowvram":
            lowvram = True
        else:
            if arg[0] != "-":
                workflow = arg
        i = i + 1

    result = make_dict()
    dict_set(result, "checkpoint", checkpoint)
    dict_set(result, "prompt", prompt)
    dict_set(result, "output", output)
    dict_set(result, "output_dir", output_dir)
    dict_set(result, "workflow", workflow)
    dict_set(result, "show_help", show_help)
    dict_set(result, "cpu", cpu)
    dict_set(result, "cuda_device", cuda_device)
    dict_set(result, "highvram", highvram)
    dict_set(result, "lowvram", lowvram)
    return result


def print_help():
    print("ComfyCLI - Stable Diffusion workflow executor")
    print("")
    print("Usage:")
    print("  comfycli [--checkpoint <path>] [--prompt <text>] [--output <path>]")
    print("  comfycli <workflow.json> [--output-dir <path>]")
    print("  comfycli --help")
    print("")
    print("Options:")
    print("  --checkpoint, --ckpt <path>     Model checkpoint path")
    print("  --prompt, -p <text>             Text prompt")
    print("  --output, -o <path>             Output image path")
    print("  --output-dir <path>             Output directory")
    print("  --cpu                           CPU mode (no GPU)")
    print("  --cuda-device <id>              CUDA device ID (default: 0)")
    print("  --highvram, --gpu-only          Keep all models on GPU")
    print("  --lowvram                       Offload models to CPU")
    print("  --help, -h                      Show this help")


def main():
    args: CliArgs = parse_args()
    if args.show_help:
        print_help()
        exit_program(0)
