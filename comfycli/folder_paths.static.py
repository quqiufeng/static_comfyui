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
