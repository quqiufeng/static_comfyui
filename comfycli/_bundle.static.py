# === comfy_types.static.py ===
class ModelType:
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8
    IMG_TO_IMG = 9
    FLOW_COSMOS = 10
    IMG_TO_IMG_FLOW = 11
    V_PREDICTION_DDPM = 12


class CLIPType:
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12
    WAN = 13
    HIDREAM = 14
    CHROMA = 15
    ACE = 16
    OMNIGEN2 = 17
    QWEN_IMAGE = 18
    HUNYUAN_IMAGE = 19
    HUNYUAN_VIDEO_15 = 20
    OVIS = 21
    KANDINSKY5 = 22
    KANDINSKY5_IMAGE = 23
    NEWBIE = 24
    FLUX2 = 25
    LONGCAT_IMAGE = 26
    COGVIDEOX = 27
    LENS = 28
    PIXELDIT = 29
    IDEOGRAM4 = 30
    BOOGU = 31
    KREA2 = 32


class TEModel:
    CLIP_L = 1
    CLIP_H = 2
    CLIP_G = 3
    T5_XXL = 4
    T5_XL = 5
    T5_BASE = 6
    LLAMA3_8 = 7
    T5_XXL_OLD = 8
    GEMMA_2_2B = 9
    QWEN25_3B = 10
    QWEN25_7B = 11
    BYT5_SMALL_GLYPH = 12
    GEMMA_3_4B = 13
    MISTRAL3_24B = 14
    MISTRAL3_24B_PRUNED_FLUX2 = 15
    QWEN3_4B = 16
    QWEN3_2B = 17
    GEMMA_3_12B = 18
    JINA_CLIP_2 = 19
    QWEN3_8B = 20
    QWEN3_06B = 21
    GEMMA_3_4B_VISION = 22
    QWEN35_08B = 23
    QWEN35_2B = 24
    QWEN35_4B = 25
    QWEN35_9B = 26
    QWEN35_27B = 27
# === folder_paths.static.py ===
supported_pt_extensions: list[str] = [".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"]

folder_names_and_paths: dict = make_dict()
filename_list_cache: dict = make_dict()
extension_mimetypes_cache: dict = make_dict()

base_path: str = ""

# Output/temp/input directories
output_directory: str = ""
temp_directory: str = ""
input_directory: str = ""
user_directory: str = ""


# Legacy folder name mapping
def map_legacy(folder_name: str) -> str:
    if folder_name == "unet":
        return "diffusion_models"
    if folder_name == "clip":
        return "text_encoders"
    return folder_name


# Path utility functions (replacements for os.path.*)
def path_join2(a: str, b: str) -> str:
    if str_ends_with(a, "/"):
        return a + b
    return a + "/" + b


def path_join3(a: str, b: str, c: str) -> str:
    return path_join2(path_join2(a, b), c)


def path_dirname(p: str) -> str:
    # Find last /, return everything before it
    pos: int = 0
    i: int = 0
    while i < str_length(p):
        if str_slice(p, i, i + 1) == "/":
            pos = i
        i = i + 1
    if pos == 0:
        return ""
    return str_slice(p, 0, pos)


def path_split(p: str) -> list[str]:
    # Split into (dir, filename)
    pos: int = 0
    i: int = 0
    while i < str_length(p):
        if str_slice(p, i, i + 1) == "/":
            pos = i
        i = i + 1
    if pos == 0:
        return ["", p]
    return [str_slice(p, 0, pos), str_slice(p, pos + 1, str_length(p))]


# Folder registration
def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: int):
    name: str = map_legacy(folder_name)
    entry = dict_get(folder_names_and_paths, name)
    if entry:
        paths = py_list_ref(entry, 0)
        exts = py_list_ref(entry, 1)
    else:
        paths = py_list()
        exts = py_list()
        dict_set(folder_names_and_paths, name, [paths, exts])
    if is_default:
        paths = py_list_append(full_folder_path, paths)
    else:
        paths = py_list_append(paths, full_folder_path)
    dict_set(folder_names_and_paths, name, [paths, exts])


def get_folder_paths(folder_name: str) -> list[str]:
    entry = dict_get(folder_names_and_paths, folder_name)
    if entry:
        return py_list_ref(entry, 0)
    return py_list()


# File search
def get_full_path(folder_name: str, filename: str) -> str:
    paths: list[str] = get_folder_paths(folder_name)
    i: int = 0
    while i < py_list_length(paths):
        dir_path: str = py_list_ref(paths, i)
        full: str = path_join2(dir_path, filename)
        if os_file_exists(full):
            return full
        i = i + 1
    return ""


def filter_files_extensions(files: list[str], extensions: list[str]) -> list[str]:
    if py_list_length(extensions) == 0:
        return files
    result: list[str] = py_list()
    i: int = 0
    while i < py_list_length(files):
        f: str = py_list_ref(files, i)
        j: int = 0
        found: bool = False
        while j < py_list_length(extensions):
            ext: str = py_list_ref(extensions, j)
            if str_ends_with(f, ext):
                found = True
            j = j + 1
        if found:
            result = py_list_append(result, f)
        i = i + 1
    return result


def get_filename_list(folder_name: str) -> list[str]:
    name: str = map_legacy(folder_name)
    entry = dict_get(folder_names_and_paths, name)
    if entry is False:
        return py_list()
    paths: list[str] = py_list_ref(entry, 0)
    exts: list[str] = py_list_ref(entry, 1)

    all_files: list[str] = py_list()
    i: int = 0
    while i < py_list_length(paths):
        dir_path: str = py_list_ref(paths, i)
        if os_file_exists(dir_path):
            files: list[str] = os_list_dir(dir_path)
            j: int = 0
            while j < py_list_length(files):
                f: str = py_list_ref(files, j)
                full: str = path_join2(dir_path, f)
                if os_file_exists(full):
                    all_files = py_list_append(all_files, f)
                j = j + 1
        i = i + 1
    return filter_files_extensions(all_files, exts)


# Output path generation
def get_save_image_path(filename_prefix: str, output_dir: str, image_width: int, image_height: int):
    # Compute variables for filename
    prefix: str = filename_prefix
    prefix = str_replace(prefix, "%width%", string_of_int(image_width))
    prefix = str_replace(prefix, "%height%", string_of_int(image_height))
    prefix = str_replace(prefix, "%year%", "0000")
    prefix = str_replace(prefix, "%month%", "00")
    prefix = str_replace(prefix, "%day%", "00")

    # Ensure output directory exists
    if os_file_exists(output_dir) is False:
        os_mkdir(output_dir)

    # Find next counter
    existing: list[str] = os_list_dir(output_dir)
    max_counter: int = 0
    i: int = 0
    while i < py_list_length(existing):
        f: str = py_list_ref(existing, i)
        if str_starts_with(f, prefix):
            rest: str = str_slice(f, str_length(prefix), str_length(f))
            num: int = string_to_int(rest)
            if num > max_counter:
                max_counter = num
        i = i + 1
    counter: int = max_counter + 1

    # Return tuple
    return [output_dir, prefix, counter, "", prefix]


# Directory management
def get_output_directory() -> str:
    return output_directory


def set_output_directory(dir: str):
    global output_directory
    output_directory = dir


def get_temp_directory() -> str:
    return temp_directory


def set_temp_directory(dir: str):
    global temp_directory
    temp_directory = dir


def get_input_directory() -> str:
    return input_directory


def set_input_directory(dir: str):
    global input_directory
    input_directory = dir


def get_user_directory() -> str:
    return user_directory


def set_user_directory(dir: str):
    global user_directory
    user_directory = dir


def get_directory_by_type(type_name: str) -> str:
    if type_name == "output":
        return output_directory
    if type_name == "temp":
        return temp_directory
    if type_name == "input":
        return input_directory
    return ""


# Annotated filepath
def annotated_filepath(name: str):
    n: int = str_length(name)
    if n > 8:
        suffix: str = str_slice(name, n - 8, n)
        if suffix == "[output]":
            base_dir: str = output_directory
            return [str_slice(name, 0, n - 8), base_dir]
        if suffix == "[input]":
            base_dir: str = input_directory
            return [str_slice(name, 0, n - 8), base_dir]
        if suffix == "[temp]":
            base_dir: str = temp_directory
            return [str_slice(name, 0, n - 8), base_dir]
    return [name, ""]


def get_annotated_filepath(name: str) -> str:
    parts = annotated_filepath(name)
    file: str = py_list_ref(parts, 0)
    base_dir: str = py_list_ref(parts, 1)
    if base_dir != "":
        return path_join2(base_dir, file)
    return file


def exists_annotated_filepath(name: str) -> bool:
    full: str = get_annotated_filepath(name)
    return os_file_exists(full)


def init(base: str):
    global base_path
    global output_directory
    global temp_directory
    global input_directory
    global user_directory

    base_path = base

    output_directory = path_join2(base_path, "output")
    temp_directory = path_join2(base_path, "temp")
    input_directory = path_join2(base_path, "input")
    user_directory = path_join2(base_path, "user")

    # Register model folders
    add_model_folder_path("checkpoints", path_join2(base_path, "models/checkpoints"), 0)
    add_model_folder_path("configs", path_join2(base_path, "models/configs"), 0)
    add_model_folder_path("loras", path_join2(base_path, "models/loras"), 0)
    add_model_folder_path("vae", path_join2(base_path, "models/vae"), 0)
    add_model_folder_path("text_encoders", path_join2(base_path, "models/text_encoders"), 0)
    add_model_folder_path("diffusion_models", path_join2(base_path, "models/unet"), 0)
    add_model_folder_path("clip_vision", path_join2(base_path, "models/clip_vision"), 0)
    add_model_folder_path("style_models", path_join2(base_path, "models/style_models"), 0)
    add_model_folder_path("embeddings", path_join2(base_path, "models/embeddings"), 0)
    add_model_folder_path("diffusers", path_join2(base_path, "models/diffusers"), 0)
    add_model_folder_path("vae_approx", path_join2(base_path, "models/vae_approx"), 0)
    add_model_folder_path("controlnet", path_join2(base_path, "models/controlnet"), 0)
    add_model_folder_path("gligen", path_join2(base_path, "models/gligen"), 0)
    add_model_folder_path("upscale_models", path_join2(base_path, "models/upscale_models"), 0)
    add_model_folder_path("hypernetworks", path_join2(base_path, "models/hypernetworks"), 0)
    add_model_folder_path("classifiers", path_join2(base_path, "models/classifiers"), 0)
    add_model_folder_path("embeddings", path_join2(base_path, "models/embeddings"), 0)

    # Create input dir
    if os_file_exists(input_directory) is False:
        os_mkdir(input_directory)


def main():
    init(os_getcwd())

# === cli_args.static.py ===
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

# === supported_models_base.static.py ===
# BAC = model config base
# Stored as list: [unet_cfg_keys, unet_cfg_vals, required_keys, latent_name, model_type_val]

def make_bac(unet_cfg_keys: list[str], unet_cfg_vals: list, required_keys: list[str], latent_name: str, model_type_val: int) -> list:
    return py_list(unet_cfg_keys, unet_cfg_vals, required_keys, latent_name, model_type_val)

def bac_unet_cfg_keys(bac: list) -> list[str]:
    return py_list_ref(bac, 0)

def bac_unet_cfg_vals(bac: list) -> list:
    return py_list_ref(bac, 1)

def bac_required_keys(bac: list) -> list[str]:
    return py_list_ref(bac, 2)

def bac_latent_format_name(bac: list) -> str:
    return py_list_ref(bac, 3)

def bac_model_type_value(bac: list) -> int:
    return py_list_ref(bac, 4)


def bac_matches(bac: list, unet_config: dict, state_dict: dict) -> bool:
    cfg_keys: list[str] = bac_unet_cfg_keys(bac)
    cfg_vals: list = bac_unet_cfg_vals(bac)
    i: int = 0
    while i < py_list_length(cfg_keys):
        k: str = py_list_ref(cfg_keys, i)
        if not dict_contains(unet_config, k):
            return False
        v = py_list_ref(cfg_vals, i)
        if dict_get(unet_config, k) != v:
            return False
        i = i + 1
    req: list[str] = bac_required_keys(bac)
    j: int = 0
    while j < py_list_length(req):
        rk: str = py_list_ref(req, j)
        if not dict_contains(state_dict, rk):
            return False
        j = j + 1
    return True
# === supported_models.static.py ===

# SDXL
SDXL = make_bac(
    make_dict_from("model_channels", 320, "use_linear_in_transformer", True, "transformer_depth", py_list(0, 0, 2, 2, 10, 10), "context_dim", 2048, "adm_in_channels", 2816, "use_temporal_attention", False),
    py_list(),
    py_list(),
    "SDXL",
    1
)

# SDXL Refiner
SDXL_REFINER = make_bac(
    make_dict_from("model_channels", 384, "use_linear_in_transformer", True, "transformer_depth", py_list(0, 0, 4, 4), "context_dim", 2560, "adm_in_channels", 2560, "use_temporal_attention", False),
    py_list(),
    py_list(),
    "SDXL",
    1
)

# SD1.5
SD15 = make_bac(
    make_dict_from("context_dim", 768, "model_channels", 320, "use_linear_in_transformer", False, "adm_in_channels", 0, "use_temporal_attention", False),
    py_list(),
    py_list(),
    "SD15",
    1
)

# SD2.0
SD20 = make_bac(
    make_dict_from("context_dim", 1024, "model_channels", 320, "use_linear_in_transformer", False, "adm_in_channels", 0, "use_temporal_attention", False),
    py_list(),
    py_list(),
    "SD15",
    1
)

# SD3
SD3 = make_bac(
    make_dict_from("in_channels", 16, "pos_embed_scaling_factor", 0),
    py_list(),
    py_list(),
    "SD3",
    6
)

# Flux
FLUX = make_bac(
    make_dict_from("image_model", "flux", "guidance_embed", True),
    py_list(),
    py_list(),
    "Flux",
    8
)

# Flux Schnell (no guidance)
FLUX_SCHNELL = make_bac(
    make_dict_from("image_model", "flux", "guidance_embed", False),
    py_list(),
    py_list(),
    "Flux",
    8
)

# All registered models
ALL_MODELS: list = py_list(SDXL, SDXL_REFINER, SD15, SD20, SD3, FLUX, FLUX_SCHNELL)


def find_model(unet_config: dict, state_dict: dict) -> list:
    i: int = 0
    while i < py_list_length(ALL_MODELS):
        m: list = py_list_ref(ALL_MODELS, i)
        if bac_matches(m, unet_config, state_dict):
            return m
        i = i + 1
    return py_list()
# === model_detection.static.py ===
# model_detection: state_dict → model architecture identification
# Uses key-PRESENCE patterns (no tensor shape inspection needed for basic detection)


def detect_unet_config(state_dict: dict, prefix: str) -> dict:
    result: dict = make_dict()

    has_double_blocks: bool = dict_contains(state_dict, prefix + "double_blocks.0.img_attn.norm.key_norm.weight")
    has_joint_blocks: bool = dict_contains(state_dict, prefix + "joint_blocks.0.context_block.attn.qkv.weight")
    has_input_blocks: bool = dict_contains(state_dict, prefix + "input_blocks.0.0.weight")

    if has_double_blocks:
        dict_set(result, "image_model", "flux")
        has_guidance: bool = dict_contains(state_dict, prefix + "guidance_in.lin.weight")
        dict_set(result, "guidance_embed", has_guidance)
    elif has_joint_blocks:
        has_pos_embed: bool = dict_contains(state_dict, prefix + "pos_embed")
        dict_set(result, "in_channels", 16)
        if has_pos_embed:
            dict_set(result, "pos_embed_scaling_factor", 0)
    elif has_input_blocks:
        has_label_emb: bool = dict_contains(state_dict, prefix + "label_emb.0.0.weight")
        if has_label_emb:
            dict_set(result, "model_channels", 320)
            dict_set(result, "use_linear_in_transformer", True)
            dict_set(result, "context_dim", 2048)
            dict_set(result, "adm_in_channels", 2816)
            dict_set(result, "use_temporal_attention", False)
        else:
            has_context_768: bool = dict_contains(state_dict, prefix + "input_blocks.0.1.transformer_blocks.0.attn2.to_k.weight")
            if has_context_768:
                dict_set(result, "context_dim", 768)
            else:
                dict_set(result, "context_dim", 1024)
            dict_set(result, "model_channels", 320)
            dict_set(result, "use_linear_in_transformer", False)
            dict_set(result, "adm_in_channels", 0)
            dict_set(result, "use_temporal_attention", False)

    return result


def model_config_from_unet(state_dict: dict, unet_config: dict, models_list: list) -> list:
    i: int = 0
    while i < py_list_length(models_list):
        m: list = py_list_ref(models_list, i)
        if bac_matches(m, unet_config, state_dict):
            return m
        i = i + 1
    return py_list()
# === model_sampling.static.py ===
# Model sampling/prediction types
# Sigma schedules handled by C++ backend (libtorch_std_helper)

class ModelSamplingType:
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    EDM = 5
    CONST = 6
    V_PREDICTION_CONTINUOUS = 7
    FLOW = 8
    X0 = 9


def sampling_type_from_model_type(model_type: int) -> int:
    if model_type == 1:
        return ModelSamplingType.EPS
    elif model_type == 2 or model_type == 3:
        return ModelSamplingType.V_PREDICTION
    elif model_type == 5:
        return ModelSamplingType.EDM
    elif model_type == 6 or model_type == 8:
        return ModelSamplingType.CONST
    elif model_type == 7:
        return ModelSamplingType.V_PREDICTION_CONTINUOUS
    elif model_type == 9 or model_type == 11:
        return ModelSamplingType.X0
    return ModelSamplingType.EPS
# === latent_formats.static.py ===
# Latent format: [scale_factor, latent_channels, name]

def make_latent_format(scale: float, channels: int, name: str) -> list:
    return py_list(scale, channels, name)

LF_SD15 = make_latent_format(0.18215, 4, "SD15")
LF_SDXL = make_latent_format(0.13025, 4, "SDXL")
LF_SD3 = make_latent_format(1.0, 16, "SD3")
LF_FLUX = make_latent_format(0.3611, 16, "Flux")
LF_FLUX2 = make_latent_format(0.3611, 16, "Flux2")
LF_SC_C = make_latent_format(1.0, 16, "StableCascade")
LF_SVD = make_latent_format(1.0, 4, "SVD")
LF_HUNYUAN_VIDEO = make_latent_format(1.0, 16, "HunyuanVideo")

def latent_format_scale(lf: list) -> float:
    return py_list_ref(lf, 0)

def latent_format_channels(lf: list) -> int:
    return py_list_ref(lf, 1)

def latent_format_name(lf: list) -> str:
    return py_list_ref(lf, 2)

def latent_process_in(latent: float, lf: list) -> float:
    return latent * latent_format_scale(lf)

def latent_process_out(latent: float, lf: list) -> float:
    return latent / latent_format_scale(lf)
# === model_base.static.py ===
# Model type constants (match comfy_types ModelType)
MODEL_TYPE_EPS: int = 1
MODEL_TYPE_V_PREDICTION: int = 2
MODEL_TYPE_FLOW: int = 6
MODEL_TYPE_FLUX: int = 8


@dataclass
class BaseModel:
    model_type: int
    latent_format_name: str
    sampling_type: int
    unet_prefix: str


def make_base_model(model_type: int, latent_format_name: str, sampling_type: int, unet_prefix: str) -> BaseModel:
    return BaseModel(model_type, latent_format_name, sampling_type, unet_prefix)


def base_model_model_type(m: BaseModel) -> int:
    return m.model_type


def base_model_latent_format_name(m: BaseModel) -> str:
    return m.latent_format_name


def base_model_sampling_type(m: BaseModel) -> int:
    return m.sampling_type


def base_model_unet_prefix(m: BaseModel) -> str:
    return m.unet_prefix
# === model_management.static.py ===
DEVICE_CPU: int = -1
DEVICE_CUDA_0: int = 0


def get_torch_device() -> int:
    return DEVICE_CUDA_0


def get_free_memory() -> int:
    return torch.cuda_get_free_memory()


def soft_empty_cache():
    torch.cuda_soft_empty_cache()


def load_model_to_device(sd_handle: ptr, device: int):
    return torch.cuda_load_model(device, sd_handle)


def unload_model_from_device(t):
    torch.cuda_unload_model(t)


def unet_offload_device() -> int:
    return DEVICE_CPU
# === sd.static.py ===
@dataclass
class ModelPatcher:
    sd_handle: ptr
    model_type: int
    load_device: int
    offload_device: int


def make_model_patcher(sd_handle: ptr, model_type: int, load_device: int, offload_device: int) -> ModelPatcher:
    return ModelPatcher(sd_handle, model_type, load_device, offload_device)


@dataclass
class CLIP:
    clip_ptr: ptr
    tokenizer_ptr: ptr


def make_clip(clip_ptr: ptr, tokenizer_ptr: ptr) -> CLIP:
    return CLIP(clip_ptr, tokenizer_ptr)


@dataclass
class VAE:
    vae_ptr: ptr


def make_vae(vae_ptr: ptr) -> VAE:
    return VAE(vae_ptr)


@dataclass
class LoadResult:
    model: ModelPatcher
    clip: CLIP
    vae: VAE


def make_load_result(model: ModelPatcher, clip: CLIP, vae: VAE) -> LoadResult:
    return LoadResult(model, clip, vae)


def build_key_dict(sd_ptr: ptr) -> dict:
    n: int = torch.safetensors_count(sd_ptr)
    d: dict = make_dict()
    i: int = 0
    while i < n:
        name = torch.safetensors_name(sd_ptr, i)
        dict_set(d, name, 1)
        i = i + 1
    return d


def load_checkpoint(ckpt_path: str) -> LoadResult:
    sd_ptr: ptr = torch.safetensors_load(ckpt_path)
    state_dict: dict = build_key_dict(sd_ptr)

    unet_prefix: str = "model.diffusion_model."
    has_unet: bool = dict_contains(state_dict, unet_prefix + "input_blocks.0.0.weight")
    if not has_unet:
        has_unet = dict_contains(state_dict, unet_prefix + "double_blocks.0.img_attn.norm.key_norm.weight")
    if not has_unet:
        has_unet = dict_contains(state_dict, unet_prefix + "joint_blocks.0.context_block.attn.qkv.weight")

    has_clip_l: bool = dict_contains(state_dict, "text_model.encoder.layers.0.layer_norm.weight")
    has_clip_g: bool = dict_contains(state_dict, "text_model.encoder.layers.30.mlp.fc1.weight")
    if not has_clip_l:
        has_clip_l = dict_contains(state_dict, "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.weight")
    if not has_clip_g:
        has_clip_g = dict_contains(state_dict, "conditioner.embedders.1.transformer.text_model.encoder.layers.30.mlp.fc1.weight")
    has_vae: bool = dict_contains(state_dict, "first_stage_model.decoder.conv_in.weight")
    if not has_vae:
        has_vae = dict_contains(state_dict, "decoder.conv_in.weight")
    if not has_vae:
        has_vae = dict_contains(state_dict, "conditioner.embedders.3.decoder.conv_in.weight")

    model_type_val: int = 0
    if has_unet:
        model_type_val = 1

    model_mp: ModelPatcher = make_model_patcher(sd_ptr, model_type_val, 0, -1)
    clip_obj = None
    vae_obj = None

    if has_clip_l or has_clip_g:
        clip_obj = make_clip(sd_ptr, sd_ptr)
    if has_vae:
        vae_obj = make_vae(sd_ptr)

    return make_load_result(model_mp, clip_obj, vae_obj)
# === lora.static.py ===
def load_lora(lora_path: str):
    return torch.safetensors_load(lora_path)


def apply_lora(unet_handle, lora_handle, strength: float):
    return torch.lora_apply(unet_handle, lora_handle, strength)


def merge_lora_into(model_handle, lora_handle) -> int:
    return torch.lora_merge_into(model_handle, lora_handle)
# === clip_model.static.py ===

CLIP_MAX_TOKEN_LENGTH = 77

CLIP_VOCAB_SIZE = 49408
CLIP_EOS_TOKEN = 49407
CLIP_SOS_TOKEN = 49406
CLIP_PAD_TOKEN_L = 49407
CLIP_PAD_TOKEN_G = 0

CLIP_L_EMBED_DIM = 768
CLIP_G_EMBED_DIM = 1280
CLIP_L_NUM_LAYERS = 12
CLIP_G_NUM_LAYERS = 32

T5_MAX_TOKEN_LENGTH = 512


def clip_tokenizer_create(vocab_path: str, merges_path: str):
    return torch.clip_tokenizer_create(vocab_path, merges_path)


def clip_tokenizer_encode(tokenizer, text: str):
    return torch.clip_tokenizer_encode(tokenizer, text)


def clip_tokenizer_free(tokenizer):
    torch.clip_tokenizer_free(tokenizer)


def clip_text_forward(clip_module, token_ids, cast_to_float16: bool):
    return torch.clip_text_forward(clip_module, token_ids, cast_to_float16)


def sdxl_dual_clip(clip_l, clip_g, token_ids):
    return torch.sdxl_dual_clip(clip_l, clip_g, token_ids)


def sdxl_get_pooled():
    return torch.sdxl_get_pooled()


def sdxl_get_pooled_l():
    return torch.sdxl_get_pooled_l()


def t5_tokenizer_create(model_path: str):
    return torch.t5_tokenizer_create(model_path)


def t5_tokenizer_encode(tokenizer, text: str, max_len: int):
    return torch.t5_tokenizer_encode(tokenizer, text, max_len)


def t5_tokenizer_free(tokenizer):
    torch.t5_tokenizer_free(tokenizer)


def encode_sd15(clip_module, tokenizer, text: str, cast_fp16: bool):
    token_ids = clip_tokenizer_encode(tokenizer, text)
    return clip_text_forward(clip_module, token_ids, cast_fp16)


def encode_sdxl(clip_l_dict, clip_g_dict, tokenizer_l, tokenizer_g, text: str):
    tokens_l = clip_tokenizer_encode(tokenizer_l, text)
    # CLIP-L: dim=768, layers=12, heads=12, ffn=3072
    emb_l = torch.clip_text_forward_from_dict(clip_l_dict, tokens_l, 768, 12, 12, 3072)
    # CLIP-G: dim=1280, layers=32, heads=20, ffn=5120
    emb_g = torch.clip_text_forward_from_dict(clip_g_dict, tokens_l, 1280, 32, 20, 5120)
    result_list = py_list(emb_l, emb_g)
    text_emb = torch.cat(result_list, 2)
    # Pooled: approximate EOS is at position 60 (after text, before padding)
    pooled = torch.narrow(emb_g, 1, 60, 1)
    pooled = torch.squeeze(pooled, 1)
    return text_emb, pooled


def encode_flux(t5_tokenizer, t5_model, clip_l_module, clip_l_tok, text: str, max_len: int):
    t5_ids = t5_tokenizer_encode(t5_tokenizer, text, max_len)
    clip_ids = clip_tokenizer_encode(clip_l_tok, text)
    t5_emb = torch.jit_forward(t5_model, t5_ids)
    clip_emb = clip_text_forward(clip_l_module, clip_ids, True)
    return t5_emb, clip_emb


def encode_t5(t5_tokenizer, t5_model, text: str, max_len: int):
    token_ids = t5_tokenizer_encode(t5_tokenizer, text, max_len)
    return torch.jit_forward(t5_model, token_ids)
# === k_diffusion/sampling.static.py ===
def get_sigmas(sampler_type: int, steps: int, sigma_min: float, sigma_max: float):
    if sampler_type == 8:
        return torch.fm_sigmas(steps, sigma_min, sigma_max)
    else:
        return torch.sampler_sigmas(steps, sigma_min, sigma_max, "karras")


def sample_step(sampler_name: str, noise_pred, x_t, sigma_t, sigma_prev, extra=None):
    if sampler_name == "euler":
        return torch.sample_euler(noise_pred, x_t, sigma_t, sigma_prev)
    elif sampler_name == "euler_ancestral":
        return torch.sample_euler_ancestral(noise_pred, x_t, sigma_t, sigma_prev)
    elif sampler_name == "ddim":
        return torch.sample_ddim(noise_pred, x_t, sigma_t, sigma_prev, 0.0)
    elif sampler_name == "dpmpp_2m":
        old_denoised = extra
        is_first = 1 if old_denoised is None else 0
        if old_denoised is None:
            old_denoised = x_t
        return torch.sample_dpmpp_2m(noise_pred, x_t, sigma_t, sigma_prev, old_denoised, is_first)
    else:
        return torch.sample_euler(noise_pred, x_t, sigma_t, sigma_prev)


def sample_flow_step(velocity, x_t, dt: float):
    return torch.fm_step(velocity, x_t, dt)
# === controlnet.static.py ===
def load_controlnet(controlnet_path: str):
    return torch.safetensors_load(controlnet_path)


def controlnet_forward(weights, input, timestep, text_emb, hint, hint_channels: int):
    return torch.controlnet_forward(weights, input, timestep, text_emb, hint, hint_channels)


def controlnet_apply(unet_features, control_features, strength: float):
    return torch.controlnet_apply(unet_features, control_features, strength)
# === nodes.static.py ===


NODE_CLASS_MAPPINGS: dict = make_dict()
NODE_DISPLAY_NAMES: dict = make_dict()


def register_node(class_type: str, display: str, func_name: str, ret_types: list,
                  is_output: bool):
    meta = make_dict()
    dict_set(meta, "display", display)
    dict_set(meta, "function", func_name)
    dict_set(meta, "return_types", ret_types)
    dict_set(meta, "output_node", is_output)
    dict_set(NODE_CLASS_MAPPINGS, class_type, meta)
    dict_set(NODE_DISPLAY_NAMES, class_type, display)


def checkpoint_loader_simple(inputs):
    ckpt_name = dict_get(inputs, "ckpt_name")
    if str_starts_with(ckpt_name, "/"):
        ckpt_path = ckpt_name
    else:
        ckpt_path = "/data/models/image/" + ckpt_name
    result = load_checkpoint(ckpt_path)
    return (result.model, result.clip, result.vae)


register_node("CheckpointLoaderSimple", "Load Checkpoint",
              "checkpoint_loader_simple", ("MODEL", "CLIP", "VAE"), False)


def dual_clip_loader(inputs):
    clip_name1 = dict_get(inputs, "clip_name1")
    clip_name2 = dict_get(inputs, "clip_name2")
    base_path = "/data/models/image/"
    clip_g_sd = torch.safetensors_load(base_path + clip_name1)
    clip_l_sd = torch.safetensors_load(base_path + clip_name2)
    result = make_dict()
    dict_set(result, "clip_g", clip_g_sd)
    dict_set(result, "clip_l", clip_l_sd)
    return (result,)


register_node("DualCLIPLoader", "Dual CLIP Loader",
              "dual_clip_loader", ("CLIP",), False)


def clip_text_encode(inputs):
    text = dict_get(inputs, "text")
    clip_obj = dict_get(inputs, "clip")
    clip_l = dict_get(clip_obj, "clip_l")
    clip_g = dict_get(clip_obj, "clip_g")
    tokenizer_l = clip_tokenizer_create("/data/models/image/clip_l_vocab.json",
                                         "/data/models/image/clip_l_merges.txt")
    tokenizer_g = clip_tokenizer_create("/data/models/image/clip_g_vocab.json",
                                         "/data/models/image/clip_g_merges.txt")
    text_emb, pooled = encode_sdxl(clip_l, clip_g, tokenizer_l, tokenizer_g, text)
    clip_tokenizer_free(tokenizer_l)
    clip_tokenizer_free(tokenizer_g)
    cond = make_dict()
    dict_set(cond, "crossattn", text_emb)
    dict_set(cond, "pooled_output", pooled)
    return (cond,)


register_node("CLIPTextEncode", "CLIP Text Encode (Prompt)",
              "clip_text_encode", ("CONDITIONING",), False)


def empty_latent_image(inputs):
    width = dict_get(inputs, "width")
    height = dict_get(inputs, "height")
    batch_size = 1
    latent = torch.zeros([batch_size, 4, height // 8, width // 8])
    result = make_dict()
    dict_set(result, "samples", latent)
    return (result,)


register_node("EmptyLatentImage", "Empty Latent Image",
              "empty_latent_image", ("LATENT",), False)


def vae_decode(inputs):
    vae_obj: VAE = dict_get(inputs, "vae")
    samples = dict_get(inputs, "samples")
    latent_tensor = dict_get(samples, "samples")
    image = torch.vae_decode_from_dict(vae_obj.vae_ptr, latent_tensor)
    return (image,)

# VAE test: bypass UNet, use correct latent from file
def vae_decode_test(inputs):
    vae_obj: VAE = dict_get(inputs, "vae")
    # Load correct latent from raw binary
    data = file_read_binary("/tmp/correct_latent.bin")
    if data is None:
        return (None,)
    # Create tensor from raw data (1,4,128,128 float16)
    latent = torch.tensor_from_blob(data, [1, 4, 128, 128], 3)  # dtype=3 = float16
    image = torch.vae_decode_from_dict(vae_obj.vae_ptr, latent)
    return (image,)

def vae_decode_debug(inputs):
    vae_obj: VAE = dict_get(inputs, "vae")
    # Load correct latent from file
    import_data = file_read_all("/tmp/correct_latent2.bin")
    if import_data is None:
        return (None,)
    # Use a bypass: just decode the native latent from KSampler
    samples = dict_get(inputs, "samples")
    latent_tensor = dict_get(samples, "samples")
    image = torch.vae_decode_from_dict(vae_obj.vae_ptr, latent_tensor)
    return (image,)


register_node("VAEDecode", "VAE Decode",
              "vae_decode", ("IMAGE",), False)


def vae_encode(inputs):
    vae: VAE = dict_get(inputs, "vae")
    pixels = dict_get(inputs, "pixels")
    tile_size = 512
    overlap = 64
    vae_ptr = vae.vae_ptr
    latent = torch.vae_encode_tiled(vae_ptr, pixels, tile_size, overlap)
    result = make_dict()
    dict_set(result, "samples", latent)
    return (result,)


register_node("VAEEncode", "VAE Encode",
              "vae_encode", ("LATENT",), False)


def k_sampler_inner(inputs):
    model: ModelPatcher = dict_get(inputs, "model")
    seed = dict_get(inputs, "seed")
    steps = dict_get(inputs, "steps")
    cfg = dict_get(inputs, "cfg")
    sampler_name = dict_get(inputs, "sampler_name")
    scheduler = dict_get(inputs, "scheduler")
    positive = dict_get(inputs, "positive")
    negative = dict_get(inputs, "negative")
    latent_image = dict_get(inputs, "latent_image")
    denoise = dict_get(inputs, "denoise")
    cond = dict_get(positive, "crossattn")
    uncond = dict_get(negative, "crossattn")
    pooled_pos = dict_get(positive, "pooled_output")
    pooled_neg = dict_get(negative, "pooled_output")
    latent_tensor = dict_get(latent_image, "samples")
    h = tensor_shape_dim(latent_tensor, 2)
    w = tensor_shape_dim(latent_tensor, 3)
    torch.manual_seed(seed)
    noise = torch.randn([1, 4, h, w])
    sigmas = torch.sampler_sigmas(steps, 0.029, 14.615, scheduler)
    noise = torch.mul(noise, torch.narrow(sigmas, 0, 0, 1))
    x = noise
    sd_handle = model.sd_handle
    n = 0
    while n < steps:
        sigma_t = torch.narrow(sigmas, 0, n, 1)
        sigma_prev = torch.narrow(sigmas, 0, n + 1, 1)
        # Pass CPU copy of sigma to UNet (needs CPU data_ptr)
        s_in = torch.to_cpu(sigma_t)
        cond_out = model_fn(sd_handle, x, s_in, cond, pooled_pos)
        uncond_out = model_fn(sd_handle, x, s_in, uncond, pooled_neg)
        eps = torch.add(uncond_out, torch.mul(torch.sub(cond_out, uncond_out), cfg))
        x = torch.add(x, torch.mul(eps, torch.sub(sigma_prev, sigma_t)))
        n = n + 1
    result = make_dict()
    dict_set(result, "samples", x)
    return (result,)


def model_fn(sd_handle, x, sigma, text_emb, pooled_emb):
    os_h = 1024.0
    os_w = 1024.0
    crop_t = 0.0
    crop_l = 0.0
    ts_h = 1024.0
    ts_w = 1024.0
    sigma_d = torch.to_cuda(sigma)
    sigma_2 = torch.mul(sigma_d, sigma_d)
    factor = torch.add(sigma_2, 1.0)
    divisor = torch.pow(factor, 0.5)
    x_scaled = torch.div(x, divisor)
    return torch.sdxl_unet_forward(sd_handle, x_scaled, sigma, text_emb, pooled_emb,
                                   os_h, os_w, crop_t, crop_l, ts_h, ts_w)


register_node("KSampler", "KSampler",
              "k_sampler_inner", ("LATENT",), False)


def save_image(inputs):
    images = dict_get(inputs, "images")
    filename_prefix = dict_get(inputs, "filename_prefix")
    output_dir = dict_get(inputs, "output_dir")
    if output_dir is None:
        output_dir = "/tmp/comfy_output"
    torch.save_image(images, output_dir + "/" + filename_prefix + ".png", 0)
    return (images,)


register_node("SaveImage", "Save Image",
              "save_image", ("IMAGE",), True)


def call_node(class_type: str, inputs):
    if class_type == "CheckpointLoaderSimple":
        return checkpoint_loader_simple(inputs)
    elif class_type == "DualCLIPLoader":
        return dual_clip_loader(inputs)
    elif class_type == "CLIPTextEncode":
        return clip_text_encode(inputs)
    elif class_type == "EmptyLatentImage":
        return empty_latent_image(inputs)
    elif class_type == "VAEDecode":
        return vae_decode(inputs)
    elif class_type == "VAEEncode":
        return vae_encode(inputs)
    elif class_type == "KSampler":
        return k_sampler_inner(inputs)
    elif class_type == "SaveImage":
        return save_image(inputs)
    else:
        return (None,)
# === execution.static.py ===

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
                    outputs = call_node(class_type, resolved)
                    dict_set(node_outputs, nid, outputs)
                    dict_set(executed, nid, 1)
                    remaining = remaining - 1
                    progress = 1
            i = i + 1
        if progress == 0:
            break
    return node_outputs
# === main.static.py ===


def guard_main():
    main()

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

