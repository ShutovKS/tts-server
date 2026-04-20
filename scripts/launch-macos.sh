#!/usr/bin/env bash
# FILE: scripts/launch-macos.sh
# VERSION: 1.0.0
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
#   MODEL_OPTIONS_DATA - Curated model menu entries mapped to runtime family contours and local artifact folders.
#   get_project_root - Resolve the repository root relative to the script location.
#   read_trimmed_input - Read one prompt and normalize surrounding whitespace.
#   read_secret_input - Read a transient secret without echoing it to the terminal.
#   select_menu_option - Prompt for a numbered menu selection and return the chosen record.
#   invoke_launcher_json - Execute the profile-aware Python launcher and capture its JSON payload.
#   assert_macos_preflight - Validate macOS host expectations, launcher package presence, and optionally install missing system dependencies via Homebrew.
#   ensure_family_environment - Create and verify the dedicated family environment through launcher create-env/check-env flows.
#   ensure_model_availability - Validate a model folder or Hugging Face snapshot layout and optionally download missing assets.
#   configure_service_environment - Apply transient QWEN_TTS_* and Telegram settings for the selected launch contour.
#   wait_http_health_check - Probe the configured HTTP server until /health/live responds or timeout elapses.
#   start_selected_service - Launch the selected adapter through the profile-aware launcher exec command.
#   main - Run the interactive launcher flow end-to-end.
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   LAST_CHANGE: [v1.0.0 - Added a thin interactive macOS launcher that mirrors the Windows guided flow on top of the shared launcher package while keeping system dependency installs opt-in through Homebrew]
# END_CHANGE_SUMMARY

set -euo pipefail

MODEL_OPTIONS_DATA=$(cat <<'EOF'
qwen-custom-17b|Qwen Custom 1.7B|qwen|Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|
qwen-design-17b|Qwen Design 1.7B|qwen|Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|
qwen-clone-17b|Qwen Clone 1.7B|qwen|Qwen3-TTS-12Hz-1.7B-Base-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|
qwen-custom-06b|Qwen Custom 0.6B|qwen|Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|
qwen-design-06b|Qwen Design 0.6B|qwen|Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|
qwen-clone-06b|Qwen Clone 0.6B|qwen|Qwen3-TTS-12Hz-0.6B-Base-8bit|huggingface|config.json;model.safetensors,model.safetensors.index.json;preprocessor_config.json;tokenizer_config.json,vocab.json|
omnivoice|OmniVoice|omnivoice|OmniVoice|huggingface|config.json;model.safetensors,model.safetensors.index.json;tokenizer_config.json,tokenizer.json;audio_tokenizer/config.json;audio_tokenizer/model.safetensors;audio_tokenizer/preprocessor_config.json|
piper-lessac|Piper en_US lessac medium|piper|Piper-en_US-lessac-medium|piper|model.onnx;model.onnx.json|en_US-lessac-medium
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

    printf '\n%s\n' "$prompt"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        count=$((count + 1))
        IFS='|' read -r _key label _rest <<<"$line"
        printf '[%d] %s\n' "$count" "$label"
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
    local piper_voice="$6"
    local models_dir="$7"

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
        local repo_id token use_token
        repo_id="$(read_trimmed_input 'Enter the Hugging Face repo ID for this model: ')"
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

configure_service_environment() {
    local project_root="$1"
    local inspect_payload="$2"
    local service_key="$3"

    export QWEN_TTS_MODELS_DIR="$project_root/.models"
    export QWEN_TTS_OUTPUTS_DIR="$project_root/.outputs"
    export QWEN_TTS_VOICES_DIR="$project_root/.voices"
    export QWEN_TTS_UPLOAD_STAGING_DIR="$project_root/.uploads"
    local selected_backend
    selected_backend="$(json_query "$inspect_payload" 'selected_backend')"
    if [[ -n "$selected_backend" ]]; then
        export QWEN_TTS_BACKEND="$selected_backend"
        export QWEN_TTS_BACKEND_AUTOSELECT=false
    fi
    export QWEN_TTS_REQUEST_TIMEOUT_SECONDS=300

    case "$service_key" in
        server)
            local bind_host bind_port
            bind_host="$(read_trimmed_input 'Host for HTTP server [0.0.0.0]: ')"
            bind_port="$(read_trimmed_input 'Port for HTTP server [8000]: ')"
            export QWEN_TTS_HOST="${bind_host:-0.0.0.0}"
            export QWEN_TTS_PORT="${bind_port:-8000}"
            export QWEN_TTS_LOG_LEVEL=info
            ;;
        telegram)
            if [[ -z "${QWEN_TTS_TELEGRAM_BOT_TOKEN:-}" ]]; then
                export QWEN_TTS_TELEGRAM_BOT_TOKEN
                QWEN_TTS_TELEGRAM_BOT_TOKEN="$(read_secret_input 'Enter Telegram bot token (it will not be saved): ')"
            fi
            if [[ -z "${QWEN_TTS_TELEGRAM_BOT_TOKEN:-}" ]]; then
                printf 'QWEN_TTS_TELEGRAM_BOT_TOKEN is required for Telegram launch.\n' >&2
                return 1
            fi
            local allowed_ids admin_ids
            allowed_ids="$(read_trimmed_input 'Allowed user IDs (comma-separated, optional): ')"
            admin_ids="$(read_trimmed_input 'Admin user IDs (comma-separated, optional): ')"
            if [[ -n "$allowed_ids" ]]; then
                export QWEN_TTS_TELEGRAM_ALLOWED_USER_IDS="$allowed_ids"
            fi
            if [[ -n "$admin_ids" ]]; then
                export QWEN_TTS_TELEGRAM_ADMIN_USER_IDS="$admin_ids"
            fi
            export QWEN_TTS_TELEGRAM_RATE_LIMIT_ENABLED=true
            export QWEN_TTS_TELEGRAM_RATE_LIMIT_PER_USER_PER_MINUTE=20
            export QWEN_TTS_TELEGRAM_DELIVERY_STORE_PATH="$project_root/.state/telegram_delivery_store.json"
            export QWEN_TTS_TELEGRAM_LOG_LEVEL=info
            ;;
        *)
            export QWEN_TTS_AUTO_PLAY_CLI=true
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
        python3.11 -m launcher --project-root "$project_root" exec --family "$family" --module "$module" &
        local server_pid=$!
        disown "$server_pid" 2>/dev/null || true
        printf 'Server process started with PID %s.\n' "$server_pid"
        wait_http_health_check "${QWEN_TTS_HOST:-0.0.0.0}" "${QWEN_TTS_PORT:-8000}" || true
        return 0
    fi

    python3.11 -m launcher --project-root "$project_root" exec --family "$family" --module "$module"
}

main() {
    local project_root
    project_root="$(get_project_root)"
    assert_macos_preflight "$project_root"

    local service_options model_record service_record service_key service_label
    service_options=$(cat <<'EOF'
server|HTTP Server
cli|CLI
telegram|Telegram Bot
EOF
)

    printf 'Interactive macOS launcher for tts-server\n'
    service_record="$(select_menu_option 'Select service to launch' "$service_options")"
    model_record="$(select_menu_option 'Select model/family to prepare' "$MODEL_OPTIONS_DATA")"

    IFS='|' read -r service_key service_label <<< "$service_record"
    local model_key model_label model_family model_folder download_strategy artifact_groups piper_voice
    IFS='|' read -r model_key model_label model_family model_folder download_strategy artifact_groups piper_voice <<< "$model_record"

    local family_python inspect_payload compatible models_dir
    family_python="$(ensure_family_environment "$project_root" "$model_family" "$service_key")"
    invoke_launcher_json "$project_root" inspect --family "$model_family" --module "$service_key"
    if [[ $LAUNCHER_JSON_EXIT_CODE -ne 0 ]]; then
        printf 'Failed to resolve launch profile:\n%s\n' "$LAUNCHER_JSON_OUTPUT" >&2
        return 1
    fi
    inspect_payload="$LAUNCHER_JSON_OUTPUT"
    compatible="$(json_query "$inspect_payload" 'compatible')"
    printf '\nSelected family: %s\n' "$model_family"
    printf 'Selected model folder: %s\n' "$model_folder"
    printf 'Host contour: %s / backend: %s\n' "$(json_query "$inspect_payload" 'host.platform_system')" "$(json_query "$inspect_payload" 'selected_backend')"
    if [[ "$compatible" != "true" ]]; then
        printf 'Selected profile is incompatible with this host even after environment setup: %s\n' "$(json_query "$inspect_payload" 'reasons')" >&2
        return 1
    fi

    models_dir="$project_root/.models"
    mkdir -p "$models_dir"
    ensure_model_availability "$family_python" "$model_label" "$model_folder" "$download_strategy" "$artifact_groups" "$piper_voice" "$models_dir"
    configure_service_environment "$project_root" "$inspect_payload" "$service_key"
    start_selected_service "$project_root" "$model_family" "$service_key" "$service_key"
}

main "$@"
