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
    width: int
    height: int
    steps: int
    cfg: float
    seed: int
    sampler: str
    scheduler: str


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
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    seed: int = 42
    sampler: str = "euler_a"
    scheduler: str = "discrete"

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
        elif arg == "--width" or arg == "-W":
            i = i + 1
            if i < argc:
                width = string_to_int(py_list_ref(args_list, i))
        elif arg == "--height" or arg == "-H":
            i = i + 1
            if i < argc:
                height = string_to_int(py_list_ref(args_list, i))
        elif arg == "--steps" or arg == "-s":
            i = i + 1
            if i < argc:
                steps = string_to_int(py_list_ref(args_list, i))
        elif arg == "--seed" or arg == "-S":
            i = i + 1
            if i < argc:
                seed = string_to_int(py_list_ref(args_list, i))
        elif arg == "--cfg" or arg == "-C":
            i = i + 1
            if i < argc:
                cfg = string_to_float(py_list_ref(args_list, i))
        elif arg == "--sampler":
            i = i + 1
            if i < argc:
                sampler = py_list_ref(args_list, i)
        elif arg == "--scheduler":
            i = i + 1
            if i < argc:
                scheduler = py_list_ref(args_list, i)
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
    dict_set(result, "width", width)
    dict_set(result, "height", height)
    dict_set(result, "steps", steps)
    dict_set(result, "cfg", cfg)
    dict_set(result, "seed", seed)
    dict_set(result, "sampler", sampler)
    dict_set(result, "scheduler", scheduler)
    return result


def print_help():
    print("ComfyCLI - Stable Diffusion workflow executor")
    print("")
    print("Usage:")
    print("  comfycli-bin [--checkpoint <path>] [--prompt <text>] [--output <path>]")
    print("  comfycli-bin <workflow.json> [--output-dir <path>]")
    print("  comfycli-bin --help")
    print("")
    print("Options:")
    print("  --checkpoint, --ckpt <path>     Model checkpoint path")
    print("  --prompt, -p <text>             Text prompt")
    print("  --output, -o <path>             Output image path")
    print("  --output-dir <path>             Output directory")
    print("  --width, -W <int>               Image width (default: 1024)")
    print("  --height, -H <int>              Image height (default: 1024)")
    print("  --steps, -s <int>                 Sampling steps (default: 20)")
    print("  --seed, -S <int>                  Random seed (default: 42)")
    print("  --cfg, -C <float>               CFG scale (default: 7.0)")
    print("  --sampler <name>                Sampler name (default: euler_a)")
    print("  --scheduler <name>              Scheduler name (default: discrete)")
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
