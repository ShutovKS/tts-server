#!/usr/bin/env bash
# FILE: scripts/launch-macos.sh
# VERSION: 1.1.6
# START_MODULE_CONTRACT
#   PURPOSE: Provide an interactive macOS launcher that orchestrates profile-aware environment setup, optional model downloads, and adapter startup.
#   SCOPE: macOS-only preflight checks, optional Homebrew-assisted system dependency installs, service/model prompts, launcher CLI orchestration, family-env bootstrap, model artifact validation, optional Hugging Face and Piper downloads, and final adapter execution.
#   DEPENDS: M-LAUNCHER, M-PROFILE-RESOLVER, M-CONFIG
#   LINKS: M-MACOS-LAUNCHER
#   ROLE: SCRIPT
#   MAP_MODE: LOCALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   FAMILY_OPTIONS_DATA - Curated family menu entries mapped to runtime contours.
#   MODEL_OPTIONS_DATA - Curated model menu entries mapped to runtime family contours and local artifact folders.
#   get_project_root - Resolve the repository root relative to the script location.
#   read_trimmed_input - Read one prompt and normalize surrounding whitespace.
#   read_secret_input - Read a transient secret without echoing it to the terminal.
#   select_menu_option - Prompt for a numbered menu selection and return the chosen record.
#   select_multiple_menu_options - Prompt for one or more numbered menu selections and return the chosen records.
#   invoke_launcher_json - Execute the profile-aware Python launcher and capture its JSON payload.
#   assert_macos_preflight - Validate macOS host expectations, launcher package presence, and optionally install missing system dependencies via Homebrew.
#   ensure_family_environment - Create and verify the dedicated family environment through launcher create-env/check-env flows.
#   ensure_model_availability - Validate a model folder or Hugging Face snapshot layout and optionally download missing assets.
#   get_model_mode - Derive the runtime lane mode for a selected model key.
#   get_runtime_capability_bindings - Derive runtime capability bindings from ensured models and selected family.
#   show_runtime_capability_bindings - Print the final runtime capability binding summary before launch.
#   configure_service_environment - Apply transient TTS_* and Telegram settings for the selected launch contour and runtime capability bindings.
#   wait_http_health_check - Probe the configured HTTP server until /health/live responds or timeout elapses.
#   http_server_pid_file_path - Resolve the repo-local PID metadata file for launcher-managed HTTP server instances.
#   load_http_server_pid_file - Load launcher-managed HTTP server PID metadata into process-local variables.
#   clear_http_server_pid_file - Remove stale or completed launcher-managed HTTP server PID metadata.
#   process_is_running - Check whether a PID is currently alive.
#   stop_http_server_pid - Gracefully stop a launcher-managed HTTP server PID and clean up when needed.
#   ensure_http_server_launch_target - Stop an existing launcher-managed server on rerun, or prompt when a foreign process occupies the target port.
#   start_selected_service - Launch the selected adapter through the profile-aware launcher exec command.
#   main - Run the interactive launcher flow end-to-end.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.1.6 - Stopped and persisted the real HTTP listener PID so launcher-managed restarts do not misclassify inherited listeners as foreign processes]
# END_CHANGE_SUMMARY

set -euo pipefail

FAMILY_OPTIONS_DATA=$(cat <<'EOF'
qwen|Qwen3
omnivoice|OmniVoice
piper|Piper
EOF
)

MODEL_OPTIONS_DATA=$(cat <<'EOF'
qwen-custom-17b|Qwen Custom 1.7B|qwen|Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice|
qwen-design-17b|Qwen Design 1.7B|qwen|Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign|
qwen-clone-17b|Qwen Clone 1.7B|qwen|Qwen3-TTS-12Hz-1.7B-Base-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|Qwen/Qwen3-TTS-12Hz-1.7B-Base|
qwen-custom-06b|Qwen Custom 0.6B|qwen|Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice|
qwen-design-06b|Qwen Design 0.6B|qwen|Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json||
qwen-clone-06b|Qwen Clone 0.6B|qwen|Qwen3-TTS-12Hz-0.6B-Base-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|Qwen/Qwen3-TTS-12Hz-0.6B-Base|
omnivoice|OmniVoice|omnivoice|OmniVoice|huggingface|config.json;model.safetensors,model.safetensors.index.json;tokenizer_config.json,tokenizer.json;audio_tokenizer/config.json;audio_tokenizer/model.safetensors;audio_tokenizer/preprocessor_config.json|k2-fsa/OmniVoice|
piper-lessac|Piper en_US lessac medium|piper|Piper-en_US-lessac-medium|piper|model.onnx;model.onnx.json||en_US-lessac-medium
EOF
)

LAUNCHER_JSON_OUTPUT=""
LAUNCHER_JSON_EXIT_CODE=0

trim_string() {
    local value="$1"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    printf '%s' "$value"
}

get_project_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    (cd "$script_dir/.." && pwd)
}

read_trimmed_input() {
    local prompt="$1"
    local value
    read -r -p "$prompt" value || value=""
    trim_string "$value"
}

read_secret_input() {
    local prompt="$1"
    local value
    read -r -s -p "$prompt" value || value=""
    printf '\n' >&2
    printf '%s' "$value"
}

prompt_yes_no() {
    local prompt="$1"
    local answer
    answer="$(read_trimmed_input "$prompt")"
    case "${answer,,}" in
        y|yes) return 0 ;;
        *) return 1 ;;
    esac
}

json_query() {
    local payload="$1"
    local expression="$2"
    printf '%s' "$payload" | python3.11 -c 'import json, sys
expr = sys.argv[1]
value = json.load(sys.stdin)
for part in expr.split("."):
    if isinstance(value, list):
        value = value[int(part)]
    else:
        value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
' "$expression"
}

json_all_values_true() {
    local payload="$1"
    printf '%s' "$payload" | python3.11 -c 'import json, sys
data = json.load(sys.stdin)
print("true" if all(bool(value) for value in data.values()) else "false")
'
}

invoke_launcher_json() {
    local project_root="$1"
    shift
    LAUNCHER_JSON_OUTPUT="$(python3.11 -m launcher --project-root "$project_root" "$@" 2>&1)"
    LAUNCHER_JSON_EXIT_CODE=$?
    if [[ -z "$LAUNCHER_JSON_OUTPUT" ]]; then
        printf 'Launcher command returned no JSON output: %s\n' "$*" >&2
        return 1
    fi
    if ! printf '%s' "$LAUNCHER_JSON_OUTPUT" | python3.11 -c 'import json, sys; json.load(sys.stdin)' >/dev/null 2>&1; then
        printf 'Failed to parse launcher JSON output for %s:\n%s\n' "$*" "$LAUNCHER_JSON_OUTPUT" >&2
        return 1
    fi
}

select_menu_option() {
    local prompt="$1"
    local options_data="$2"
    local choice=0
    local count=0
    local line

    printf '\n%s\n' "$prompt" >&2
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        count=$((count + 1))
        IFS='|' read -r _key label _rest <<<"$line"
        printf '[%d] %s\n' "$count" "$label" >&2
    done <<< "$options_data"

    while true; do
        local raw
        raw="$(read_trimmed_input 'Enter option number: ')"
        if [[ "$raw" =~ ^[0-9]+$ ]] && (( raw >= 1 && raw <= count )); then
            choice=0
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue
                choice=$((choice + 1))
                if (( choice == raw )); then
                    printf '%s' "$line"
                    return 0
                fi
            done <<< "$options_data"
        fi
        printf 'Enter a number from the menu.\n' >&2
    done
}

select_multiple_menu_options() {
    local prompt="$1"
    local options_data="$2"
    local count=0
    local line
    local raw
    local selection_output=""

    printf '\n%s\n' "$prompt" >&2
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        count=$((count + 1))
        IFS='|' read -r _key label _rest <<<"$line"
        printf '[%d] %s\n' "$count" "$label" >&2
    done <<< "$options_data"

    if (( count == 0 )); then
        printf 'No options are available for selection.\n' >&2
        return 1
    fi

    while true; do
        local valid="true"
        local selected_indexes=()
        raw="$(read_trimmed_input 'Enter one or more option numbers separated by comma: ')"
        if [[ -z "$raw" ]]; then
            printf 'Select at least one option from the menu.\n' >&2
            continue
        fi

        local old_ifs="$IFS"
        IFS=','
        read -r -a tokens <<< "$raw"
        IFS="$old_ifs"
        for token in "${tokens[@]}"; do
            local trimmed choice duplicate="false"
            trimmed="$(trim_string "$token")"
            if [[ ! "$trimmed" =~ ^[0-9]+$ ]] || (( trimmed < 1 || trimmed > count )); then
                valid="false"
                break
            fi
            for choice in "${selected_indexes[@]:-}"; do
                if [[ "$choice" == "$trimmed" ]]; then
                    duplicate="true"
                    break
                fi
            done
            if [[ "$duplicate" == "false" ]]; then
                selected_indexes+=("$trimmed")
            fi
        done

        if [[ "$valid" != "true" || ${#selected_indexes[@]} -eq 0 ]]; then
            printf 'Enter one or more valid menu numbers separated by commas.\n' >&2
            continue
        fi

        selection_output=""
        local position=0
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            position=$((position + 1))
            for choice in "${selected_indexes[@]}"; do
                if (( position == choice )); then
                    if [[ -n "$selection_output" ]]; then
                        selection_output+=$'\n'
                    fi
                    selection_output+="$line"
                    break
                fi
            done
        done <<< "$options_data"

        if [[ -n "$selection_output" ]]; then
            printf '%s\n' "$selection_output"
            return 0
        fi
    done
}

resolve_http_probe_host() {
    local bind_host
    bind_host="$(trim_string "$1")"
    case "${bind_host,,}" in
        ""|0.0.0.0|::|[::]) printf '127.0.0.1' ;;
        *) printf '%s' "$bind_host" ;;
    esac
}

offer_brew_installs() {
    local missing_python="$1"
    local missing_ffmpeg="$2"
    local formulas=()

    if ! command -v brew >/dev/null 2>&1; then
        printf 'Homebrew was not found. Install it first from https://brew.sh and then rerun this launcher.\n' >&2
        return 1
    fi

    if [[ "$missing_python" == "true" ]]; then
        formulas+=(python@3.11)
    fi
    if [[ "$missing_ffmpeg" == "true" ]]; then
        formulas+=(ffmpeg)
    fi
    if [[ ${#formulas[@]} -eq 0 ]]; then
        return 0
    fi

    printf 'Missing macOS system dependencies detected.\n' >&2
    printf 'The launcher can install them with Homebrew: brew install %s\n' "${formulas[*]}" >&2
    if prompt_yes_no 'Run brew install now? [y/N]: '; then
        brew install "${formulas[@]}"
    else
        printf 'Launch cancelled until the required system dependencies are installed.\n' >&2
        return 1
    fi
}

assert_macos_preflight() {
    local project_root="$1"
    local system_name="$(uname -s)"
    local missing_python="false"
    local missing_ffmpeg="false"

    if [[ "$system_name" != "Darwin" ]]; then
        printf 'This launcher supports macOS only.\n' >&2
        return 1
    fi
    if [[ ! -d "$project_root/launcher" ]]; then
        printf 'Launcher package was not found under project root: %s\n' "$project_root" >&2
        return 1
    fi
    if ! command -v python3.11 >/dev/null 2>&1; then
        missing_python="true"
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
        missing_ffmpeg="true"
    fi

    if [[ "$missing_python" == "true" || "$missing_ffmpeg" == "true" ]]; then
        offer_brew_installs "$missing_python" "$missing_ffmpeg"
    fi

    if ! command -v python3.11 >/dev/null 2>&1; then
        printf "The 'python3.11' command was not found. Install python@3.11 with Homebrew and retry.\n" >&2
        return 1
    fi
    if ! python3.11 --version >/dev/null 2>&1; then
        printf "Python 3.11 was not found through 'python3.11'.\n" >&2
        return 1
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
        printf 'ffmpeg was not found in PATH. Install ffmpeg and retry.\n' >&2
        return 1
    fi
}

collect_validation_roots() {
    local model_root="$1"
    [[ -d "$model_root" ]] || return 0
    printf '%s\n' "$model_root"
    local snapshot_root
    for snapshot_root in "$model_root"/snapshots/*; do
        [[ -d "$snapshot_root" ]] || continue
        printf '%s\n' "$snapshot_root"
    done
}

artifact_group_exists() {
    local root_path="$1"
    local group="$2"
    local candidate
    local old_ifs="$IFS"
    IFS=','
    read -r -a candidates <<< "$group"
    IFS="$old_ifs"
    for candidate in "${candidates[@]}"; do
        if [[ -e "$root_path/$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

test_model_artifacts() {
    local model_folder="$1"
    local artifact_groups="$2"
    local models_dir="$3"
    local model_root="$models_dir/$model_folder"
    local root
    local old_ifs="$IFS"
    local group
    IFS=';'
    read -r -a groups <<< "$artifact_groups"
    IFS="$old_ifs"

    while IFS= read -r root; do
        [[ -n "$root" ]] || continue
        local all_groups_satisfied="true"
        for group in "${groups[@]}"; do
            if ! artifact_group_exists "$root" "$group"; then
                all_groups_satisfied="false"
                break
            fi
        done
        if [[ "$all_groups_satisfied" == "true" ]]; then
            printf 'true|%s|%s' "$root" "$model_root"
            return 0
        fi
    done < <(collect_validation_roots "$model_root")

    printf 'false||%s' "$model_root"
}

ensure_family_environment() {
    local project_root="$1"
    local family="$2"
    local module="$3"

    invoke_launcher_json "$project_root" create-env --family "$family" --module "$module" --apply
    if [[ $LAUNCHER_JSON_EXIT_CODE -ne 0 ]]; then
        printf 'Failed to prepare the family environment for %s:\n%s\n' "$family" "$LAUNCHER_JSON_OUTPUT" >&2
        return 1
    fi

    invoke_launcher_json "$project_root" check-env --family "$family" --module "$module"
    if [[ $LAUNCHER_JSON_EXIT_CODE -ne 0 ]]; then
        printf 'Environment check for %s failed:\n%s\n' "$family" "$LAUNCHER_JSON_OUTPUT" >&2
        return 1
    fi

    local import_returncode
    import_returncode="$(json_query "$LAUNCHER_JSON_OUTPUT" 'check_env.import_check.returncode')"
    if [[ "$import_returncode" != "0" ]]; then
        printf 'Runtime import check for %s failed:\n%s\n' "$family" "$LAUNCHER_JSON_OUTPUT" >&2
        return 1
    fi
    if [[ "$(json_query "$LAUNCHER_JSON_OUTPUT" 'check_env.import_check.stdout' | json_all_values_true)" != "true" ]]; then
        printf 'Environment %s is missing one or more required runtime imports.\n' "$family" >&2
        return 1
    fi

    json_query "$LAUNCHER_JSON_OUTPUT" 'check_env.expected_python_path'
}

invoke_huggingface_download() {
    local python_path="$1"
    local repo_id="$2"
    local target_dir="$3"
    local token="$4"

    mkdir -p "$target_dir"
    if [[ -n "$token" ]]; then
        HF_TOKEN="$token" "$python_path" -c 'from huggingface_hub import snapshot_download; import os, sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], token=os.environ.get("HF_TOKEN") or None)' "$repo_id" "$target_dir"
    else
        "$python_path" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], token=None)' "$repo_id" "$target_dir"
    fi
}

invoke_piper_download() {
    local python_path="$1"
    local piper_voice="$2"
    local target_dir="$3"

    mkdir -p "$target_dir"
    "$python_path" -m piper.download_voices "$piper_voice" --download-dir "$target_dir"
    if [[ -f "$target_dir/$piper_voice.onnx" ]]; then
        mv "$target_dir/$piper_voice.onnx" "$target_dir/model.onnx"
    fi
    if [[ -f "$target_dir/$piper_voice.onnx.json" ]]; then
        mv "$target_dir/$piper_voice.onnx.json" "$target_dir/model.onnx.json"
    fi
}

ensure_model_availability() {
    local python_path="$1"
    local label="$2"
    local folder="$3"
    local download_strategy="$4"
    local artifact_groups="$5"
    local repo_id="$6"
    local piper_voice="$7"
    local models_dir="$8"

    local validation
    validation="$(test_model_artifacts "$folder" "$artifact_groups" "$models_dir")"
    IFS='|' read -r available resolved_path expected_path <<< "$validation"
    if [[ "$available" == "true" ]]; then
        printf 'Model found: %s\n' "$resolved_path"
        return 0
    fi

    printf 'Model %s is missing or incomplete. Expected path: %s\n' "$label" "$expected_path" >&2
    if ! prompt_yes_no 'Download the model now? [y/N]: '; then
        printf 'Launch cancelled: model is not prepared locally.\n' >&2
        return 1
    fi

    if [[ "$download_strategy" == "piper" ]]; then
        invoke_piper_download "$python_path" "$piper_voice" "$expected_path"
    elif [[ "$download_strategy" == "huggingface" ]]; then
        local token use_token
        if [[ -n "$repo_id" ]]; then
            printf 'Using built-in Hugging Face repo ID for %s: %s\n' "$label" "$repo_id"
        else
            repo_id="$(read_trimmed_input 'Enter the Hugging Face repo ID for this model: ')"
        fi
        if [[ -z "$repo_id" ]]; then
            printf 'A Hugging Face repo ID is required for this download.\n' >&2
            return 1
        fi
        token=""
        use_token="$(read_trimmed_input 'Use a temporary HF token for this download? [y/N]: ')"
        case "${use_token,,}" in
            y|yes) token="$(read_secret_input 'Enter HF token (it will not be saved): ')" ;;
        esac
        invoke_huggingface_download "$python_path" "$repo_id" "$expected_path" "$token"
    else
        printf 'Unknown download strategy: %s\n' "$download_strategy" >&2
        return 1
    fi

    validation="$(test_model_artifacts "$folder" "$artifact_groups" "$models_dir")"
    IFS='|' read -r available resolved_path expected_path <<< "$validation"
    if [[ "$available" != "true" ]]; then
        printf 'Download finished, but required artifacts for %s are still missing.\n' "$label" >&2
        return 1
    fi
    printf 'Model is ready: %s\n' "$resolved_path"
}

get_model_mode() {
    local model_key="$1"
    case "$model_key" in
        qwen-custom-*|piper-*) printf 'custom' ;;
        qwen-design-*) printf 'design' ;;
        qwen-clone-*) printf 'clone' ;;
        omnivoice) printf 'all' ;;
        *) printf '' ;;
    esac
}

get_runtime_capability_bindings() {
    local family="$1"
    shift
    local custom_model=""
    local design_model=""
    local clone_model=""
    local model_record model_key folder inferred_mode

    for model_record in "$@"; do
        IFS='|' read -r model_key _model_label _model_family folder _download_strategy _artifact_groups _maybe_voice <<< "$model_record"
        inferred_mode="$(get_model_mode "$model_key")"

        if [[ "$inferred_mode" == "all" ]]; then
            custom_model="$folder"
            design_model="$folder"
            clone_model="$folder"
            continue
        fi
        if [[ "$inferred_mode" == "custom" ]]; then
            custom_model="$folder"
        elif [[ "$inferred_mode" == "design" ]]; then
            design_model="$folder"
        elif [[ "$inferred_mode" == "clone" ]]; then
            clone_model="$folder"
        fi
    done

    printf '%s|%s|%s|%s' "$family" "$custom_model" "$design_model" "$clone_model"
}

show_runtime_capability_bindings() {
    local family="$1"
    local custom_model="$2"
    local design_model="$3"
    local clone_model="$4"

    printf '\nRuntime capability bindings:\n'
    printf '  TTS_ACTIVE_FAMILY=%s\n' "$family"
    printf '  TTS_DEFAULT_CUSTOM_MODEL=%s\n' "${custom_model:-<unbound>}"
    printf '  TTS_DEFAULT_DESIGN_MODEL=%s\n' "${design_model:-<unbound>}"
    printf '  TTS_DEFAULT_CLONE_MODEL=%s\n' "${clone_model:-<unbound>}"
}

configure_service_environment() {
    local project_root="$1"
    local inspect_payload="$2"
    local service_key="$3"
    local bindings_family="$4"
    local custom_model="$5"
    local design_model="$6"
    local clone_model="$7"

    export TTS_MODELS_DIR="$project_root/.models"
    export TTS_OUTPUTS_DIR="$project_root/.outputs"
    export TTS_VOICES_DIR="$project_root/.voices"
    export TTS_UPLOAD_STAGING_DIR="$project_root/.uploads"
    export TTS_ACTIVE_FAMILY="$bindings_family"
    if [[ -n "$custom_model" ]]; then
        export TTS_DEFAULT_CUSTOM_MODEL="$custom_model"
    else
        unset TTS_DEFAULT_CUSTOM_MODEL 2>/dev/null || true
    fi
    if [[ -n "$design_model" ]]; then
        export TTS_DEFAULT_DESIGN_MODEL="$design_model"
    else
        unset TTS_DEFAULT_DESIGN_MODEL 2>/dev/null || true
    fi
    if [[ -n "$clone_model" ]]; then
        export TTS_DEFAULT_CLONE_MODEL="$clone_model"
    else
        unset TTS_DEFAULT_CLONE_MODEL 2>/dev/null || true
    fi
    local selected_backend
    selected_backend="$(json_query "$inspect_payload" 'selected_backend')"
    if [[ -n "$selected_backend" ]]; then
        export TTS_BACKEND="$selected_backend"
        export TTS_BACKEND_AUTOSELECT=false
    fi
    export TTS_REQUEST_TIMEOUT_SECONDS=300

    case "$service_key" in
        server)
            local bind_host bind_port
            bind_host="$(read_trimmed_input 'Host for HTTP server [0.0.0.0]: ')"
            bind_port="$(read_trimmed_input 'Port for HTTP server [8000]: ')"
            export TTS_HOST="${bind_host:-0.0.0.0}"
            export TTS_PORT="${bind_port:-8000}"
            export TTS_LOG_LEVEL=info
            ;;
        telegram)
            if [[ -z "${TTS_TELEGRAM_BOT_TOKEN:-}" ]]; then
                export TTS_TELEGRAM_BOT_TOKEN
                TTS_TELEGRAM_BOT_TOKEN="$(read_secret_input 'Enter Telegram bot token (it will not be saved): ')"
            fi
            if [[ -z "${TTS_TELEGRAM_BOT_TOKEN:-}" ]]; then
                printf 'TTS_TELEGRAM_BOT_TOKEN is required for Telegram launch.\n' >&2
                return 1
            fi
            local allowed_ids admin_ids
            allowed_ids="$(read_trimmed_input 'Allowed user IDs (comma-separated, optional): ')"
            admin_ids="$(read_trimmed_input 'Admin user IDs (comma-separated, optional): ')"
            if [[ -n "$allowed_ids" ]]; then
                export TTS_TELEGRAM_ALLOWED_USER_IDS="$allowed_ids"
            fi
            if [[ -n "$admin_ids" ]]; then
                export TTS_TELEGRAM_ADMIN_USER_IDS="$admin_ids"
            fi
            export TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
            export TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
            export TTS_TELEGRAM_DELIVERY_STORE_PATH="$project_root/.state/telegram_delivery_store.json"
            export TTS_TELEGRAM_LOG_LEVEL=info
            ;;
        *)
            export TTS_AUTO_PLAY_CLI=true
            ;;
    esac
}

wait_http_health_check() {
    local bind_host="$1"
    local bind_port="$2"
    local timeout_seconds="${3:-30}"
    local probe_host
    probe_host="$(resolve_http_probe_host "$bind_host")"

    python3.11 -c 'import sys, time, urllib.request
host, port, timeout_seconds = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f"http://{host}:{port}/health/live"
deadline = time.time() + timeout_seconds
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if 200 <= response.status < 300:
                print(f"HTTP server is live: {url}")
                raise SystemExit(0)
    except Exception:
        time.sleep(1)
print(f"HTTP server did not report ready at {url} within {timeout_seconds} seconds.", file=sys.stderr)
raise SystemExit(1)
' "$probe_host" "$bind_port" "$timeout_seconds"
}

assert_http_server_port_available() {
    local bind_host="$1"
    local bind_port="$2"
    local probe_host
    probe_host="$(resolve_http_probe_host "$bind_host")"

    python3.11 -c 'import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    if sock.connect_ex((host, port)) == 0:
        print(f"HTTP server port is already in use at {host}:{port}.", file=sys.stderr)
        raise SystemExit(1)
raise SystemExit(0)
' "$probe_host" "$bind_port"
}

http_server_pid_file_path() {
    local project_root="$1"
    printf '%s/.state/launcher/http-server.pid' "$project_root"
}

HTTP_SERVER_PID=""
HTTP_SERVER_BIND_HOST=""
HTTP_SERVER_BIND_PORT=""
HTTP_SERVER_FAMILY=""
HTTP_SERVER_MODULE=""

load_http_server_pid_file() {
    local pid_file="$1"
    HTTP_SERVER_PID=""
    HTTP_SERVER_BIND_HOST=""
    HTTP_SERVER_BIND_PORT=""
    HTTP_SERVER_FAMILY=""
    HTTP_SERVER_MODULE=""
    [[ -f "$pid_file" ]] || return 1

    local key value
    while IFS='=' read -r key value; do
        case "$key" in
            PID) HTTP_SERVER_PID="$value" ;;
            BIND_HOST) HTTP_SERVER_BIND_HOST="$value" ;;
            BIND_PORT) HTTP_SERVER_BIND_PORT="$value" ;;
            FAMILY) HTTP_SERVER_FAMILY="$value" ;;
            MODULE) HTTP_SERVER_MODULE="$value" ;;
        esac
    done < "$pid_file"

    [[ -n "$HTTP_SERVER_PID" ]]
}

clear_http_server_pid_file() {
    local project_root="$1"
    local pid_file
    pid_file="$(http_server_pid_file_path "$project_root")"
    rm -f "$pid_file"
}

process_is_running() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

resolve_http_server_port_owner_pid() {
    local bind_host="$1"
    local bind_port="$2"
    local probe_host
    probe_host="$(resolve_http_probe_host "$bind_host")"

    python3.11 -c 'import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    if sock.connect_ex((host, port)) != 0:
        raise SystemExit(1)
' "$probe_host" "$bind_port" >/dev/null 2>&1 || return 1

    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$bind_port" -sTCP:LISTEN 2>/dev/null | python3.11 -c 'import sys
for line in sys.stdin:
    value = line.strip()
    if value:
        print(value)
        raise SystemExit(0)
raise SystemExit(1)'
        return $?
    fi

    return 1
}

stop_http_server_pid() {
    local pid="$1"
    local project_root="$2"
    local wait_seconds="${3:-10}"
    local elapsed=0

    if ! process_is_running "$pid"; then
        clear_http_server_pid_file "$project_root"
        return 0
    fi

    kill "$pid" 2>/dev/null || true
    while process_is_running "$pid" && (( elapsed < wait_seconds )); do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if process_is_running "$pid"; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    clear_http_server_pid_file "$project_root"
}

stop_http_server_listener_pid() {
    local bind_host="$1"
    local bind_port="$2"
    local project_root="$3"
    local exclude_pid="${4:-}"
    local listener_pid

    listener_pid="$(resolve_http_server_port_owner_pid "$bind_host" "$bind_port" 2>/dev/null || true)"
    if [[ -z "$listener_pid" || "$listener_pid" == "${exclude_pid:-}" ]]; then
        return 0
    fi

    printf 'Stopping existing launcher-managed HTTP listener (PID %s) on %s:%s.\n' "$listener_pid" "$(resolve_http_probe_host "$bind_host")" "$bind_port"
    stop_http_server_pid "$listener_pid" "$project_root"
}

ensure_http_server_launch_target() {
    local project_root="$1"
    local bind_host="$2"
    local bind_port="$3"
    local family="$4"
    local module="$5"
    local pid_file
    pid_file="$(http_server_pid_file_path "$project_root")"

    if load_http_server_pid_file "$pid_file"; then
        if process_is_running "$HTTP_SERVER_PID"; then
            while true; do
                local managed_decision
                read -r -p 'Launcher-managed HTTP server is already running. [R]estart / [K]eep existing / [C]hange port: ' managed_decision || managed_decision=""
                managed_decision="$(trim_string "$managed_decision")"
                case "${managed_decision,,}" in
                    r|restart)
                        printf 'Stopping existing launcher-managed HTTP server (PID %s) before restart.\n' "$HTTP_SERVER_PID"
                        stop_http_server_pid "$HTTP_SERVER_PID" "$project_root"
                        stop_http_server_listener_pid "$bind_host" "$bind_port" "$project_root" "$HTTP_SERVER_PID"
                        break
                        ;;
                    c|change)
                        local managed_replacement_port
                        managed_replacement_port="$(read_trimmed_input 'New port for HTTP server: ')"
                        if [[ -z "$managed_replacement_port" ]]; then
                            printf 'A new port is required to continue.\n' >&2
                            continue
                        fi
                        export TTS_PORT="$managed_replacement_port"
                        return 0
                        ;;
                    k|keep|'')
                        printf 'Launch cancelled: existing launcher-managed HTTP server remains active on %s:%s.\n' "$(resolve_http_probe_host "$bind_host")" "$bind_port" >&2
                        return 1
                        ;;
                    *)
                        printf 'Enter R to restart, K to keep the existing server, or C to choose a new port.\n' >&2
                        ;;
                esac
            done
        else
            clear_http_server_pid_file "$project_root"
        fi
    fi

    if assert_http_server_port_available "$bind_host" "$bind_port"; then
        return 0
    fi

    while true; do
        local decision
        read -r -p 'Port is occupied by a non-launcher process. [S]top and restart / [K]eep existing / [C]hange port: ' decision || decision=""
        decision="$(trim_string "$decision")"
        case "${decision,,}" in
            s|stop)
                local owner_pid
                owner_pid="$(resolve_http_server_port_owner_pid "$bind_host" "$bind_port" 2>/dev/null || true)"
                printf 'Force-stop requested for the process currently occupying %s:%s.\n' "$(resolve_http_probe_host "$bind_host")" "$bind_port"
                if python3.11 -c 'import os, signal, sys
target = sys.argv[1]
if not target or target == "0":
    raise SystemExit(1)
os.kill(int(target), signal.SIGTERM)
' "$owner_pid" 2>/dev/null; then
                    sleep 1
                    if assert_http_server_port_available "$bind_host" "$bind_port"; then
                        return 0
                    fi
                    printf 'The foreign process did not release the port cleanly.\n' >&2
                else
                    printf 'Unable to stop the process currently occupying %s:%s.\n' "$(resolve_http_probe_host "$bind_host")" "$bind_port" >&2
                fi
                ;;
            c|change)
                local replacement_port
                replacement_port="$(read_trimmed_input 'New port for HTTP server: ')"
                if [[ -z "$replacement_port" ]]; then
                    printf 'A new port is required to continue.\n' >&2
                    continue
                fi
                export TTS_PORT="$replacement_port"
                if assert_http_server_port_available "$bind_host" "$replacement_port"; then
                    return 0
                fi
                ;;
            k|keep|'')
                printf 'Launch cancelled: existing non-launcher HTTP server remains active on %s:%s.\n' "$(resolve_http_probe_host "$bind_host")" "$bind_port" >&2
                return 1
                ;;
            *)
                printf 'Enter S to stop and restart, K to keep the existing server, or C to choose a new port.\n' >&2
                ;;
        esac
    done
}

start_selected_service() {
    local project_root="$1"
    local family="$2"
    local module="$3"
    local service_key="$4"

    invoke_launcher_json "$project_root" exec --family "$family" --module "$module" --dry-run
    local launch_command
    launch_command="$(json_query "$LAUNCHER_JSON_OUTPUT" 'exec.command')"
    printf '\nLaunching: %s\n' "$launch_command"

    if [[ "$service_key" == "server" ]]; then
        ensure_http_server_launch_target "$project_root" "${TTS_HOST:-0.0.0.0}" "${TTS_PORT:-8000}" "$family" "$module"
        local server_pid
        python3.11 -m launcher --project-root "$project_root" exec --family "$family" --module "$module" >/dev/null 2>&1 &
        server_pid=$!
        disown "$server_pid" 2>/dev/null || true
        printf 'Server process started with PID %s.\n' "$server_pid"
        if ! wait_http_health_check "${TTS_HOST:-0.0.0.0}" "${TTS_PORT:-8000}"; then
            stop_http_server_pid "$server_pid" "$project_root" 2>/dev/null || true
            return 1
        fi
        local listener_pid
        listener_pid="$(resolve_http_server_port_owner_pid "${TTS_HOST:-0.0.0.0}" "${TTS_PORT:-8000}" 2>/dev/null || true)"
        if [[ -n "$listener_pid" ]]; then
            server_pid="$listener_pid"
        fi
        mkdir -p "$project_root/.state/launcher"
        cat > "$(http_server_pid_file_path "$project_root")" <<EOF
PID=$server_pid
BIND_HOST=${TTS_HOST:-0.0.0.0}
BIND_PORT=${TTS_PORT:-8000}
FAMILY=$family
MODULE=$module
EOF
        return 0
    fi

    python3.11 -m launcher --project-root "$project_root" exec --family "$family" --module "$module"
}

main() {
    local project_root
    project_root="$(get_project_root)"
    assert_macos_preflight "$project_root"

    local service_options family_record service_record service_key family_key family_label family_models model_record
    service_options=$(cat <<'EOF'
server|HTTP Server
cli|CLI
telegram|Telegram Bot
EOF
)

    printf 'Interactive macOS launcher for tts-server\n'
    service_record="$(select_menu_option 'Select service to launch' "$service_options")"
    family_record="$(select_menu_option 'Select family to prepare' "$FAMILY_OPTIONS_DATA")"

    IFS='|' read -r service_key _service_label <<< "$service_record"
    IFS='|' read -r family_key family_label <<< "$family_record"

    family_models="$(printf '%s\n' "$MODEL_OPTIONS_DATA" | while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        IFS='|' read -r _model_key _model_label model_family _rest <<< "$line"
        if [[ "$model_family" == "$family_key" ]]; then
            printf '%s\n' "$line"
        fi
    done)"

    if [[ -z "$family_models" ]]; then
        printf 'No model options are configured for family %s.\n' "$family_key" >&2
        return 1
    fi

    local models_to_ensure=()
    local family_model_count=0
    while IFS= read -r model_record; do
        [[ -z "$model_record" ]] && continue
        family_model_count=$((family_model_count + 1))
    done <<< "$family_models"

    if (( family_model_count == 1 )); then
        models_to_ensure=("$family_models")
        IFS='|' read -r _single_key single_label _single_family _single_folder _single_strategy _single_groups _single_repo_id _single_voice <<< "$family_models"
        printf '\nAutomatically preparing the only model for %s: %s\n' "$family_label" "$single_label"
    else
        while IFS= read -r model_record; do
            [[ -z "$model_record" ]] && continue
            models_to_ensure+=("$model_record")
        done < <(select_multiple_menu_options 'Select models to download if missing (multiple options can be selected, separated by comma)' "$family_models")
    fi

    local family_python inspect_payload compatible models_dir runtime_bindings bindings_family custom_model design_model clone_model
    family_python="$(ensure_family_environment "$project_root" "$family_key" "$service_key")"
    invoke_launcher_json "$project_root" inspect --family "$family_key" --module "$service_key"
    if [[ $LAUNCHER_JSON_EXIT_CODE -ne 0 ]]; then
        printf 'Failed to resolve launch profile:\n%s\n' "$LAUNCHER_JSON_OUTPUT" >&2
        return 1
    fi
    inspect_payload="$LAUNCHER_JSON_OUTPUT"
    compatible="$(json_query "$inspect_payload" 'compatible')"
    printf '\nSelected family: %s\n' "$family_label"
    printf 'Host contour: %s / backend: %s\n' "$(json_query "$inspect_payload" 'host.platform_system')" "$(json_query "$inspect_payload" 'selected_backend')"
    if [[ "$compatible" != "true" ]]; then
        printf 'Selected profile is incompatible with this host even after environment setup: %s\n' "$(json_query "$inspect_payload" 'reasons')" >&2
        return 1
    fi

    models_dir="$project_root/.models"
    mkdir -p "$models_dir"
    for model_record in "${models_to_ensure[@]}"; do
        local model_key model_label model_family model_folder download_strategy artifact_groups repo_id piper_voice
        IFS='|' read -r model_key model_label model_family model_folder download_strategy artifact_groups repo_id piper_voice <<< "$model_record"
        printf 'Ensuring model: %s\n' "$model_label"
        ensure_model_availability "$family_python" "$model_label" "$model_folder" "$download_strategy" "$artifact_groups" "$repo_id" "$piper_voice" "$models_dir"
    done

    runtime_bindings="$(get_runtime_capability_bindings "$family_key" "${models_to_ensure[@]}")"
    IFS='|' read -r bindings_family custom_model design_model clone_model <<< "$runtime_bindings"
    show_runtime_capability_bindings "$bindings_family" "$custom_model" "$design_model" "$clone_model"
    configure_service_environment "$project_root" "$inspect_payload" "$service_key" "$bindings_family" "$custom_model" "$design_model" "$clone_model"
    start_selected_service "$project_root" "$family_key" "$service_key" "$service_key"
}

main "$@"
