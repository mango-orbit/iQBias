#!/usr/bin/env bash
#
# Single entry point for the query/value refinement recipe.
#
#   bash run_train.sh --dataset cub200
#   bash run_train.sh --dataset cars196 --gpu 1
#   bash run_train.sh --config configs/pcqr_cub200_arm_vpt_qv.yaml
#   bash run_train.sh --dataset cub200 --epochs 1 --name smoke_cub200
#   bash run_train.sh --dataset cub200 --dry-run
#
# The dataset root must contain one directory per dataset name:
#   <source_path>/cub200/images/...
#   <source_path>/cars196/images/...
#   <source_path>/sop/{images,Info_Files}/...
#
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The objective registry enumerates ./criteria relative to the working
# directory, so training must start from the project root.
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
CONFIG="configs/pcqr_cub200.yaml"
NAME=""
DRY=0
SETS=()

print_usage() {
    printf '%s\n' \
        "Usage: bash run_train.sh [options]" \
        "" \
        "  --dataset NAME      cub200 (default) | cars196 | sop" \
        "  --config PATH       explicit arm config; overrides --dataset" \
        "  --gpu N             GPU index (default 0, or \$GPU)" \
        "  --epochs N          override n_epochs" \
        "  --name NAME         run name (default: the config file name)" \
        "  --source-path P     dataset root (default: data)" \
        "  --set KEY=VALUE     override any config key (repeatable)" \
        "  --python PATH       interpreter (default: \$PYTHON_BIN or python)" \
        "  --dry-run           validate and print the effective config only" \
        "  -h, --help          this message" \
        "" \
        "Configs shipped in configs/:" \
        "  pcqr_cub200.yaml / pcqr_cars196.yaml / pcqr_sop.yaml   main variant" \
        "  pcqr_cub200_pcqr_static_qv.yaml                        static Q+V, no residual" \
        "  pcqr_cub200_baseline_linear.yaml                       frozen backbone" \
        "  pcqr_cub200_baseline_all_bias_bitfit.yaml              all-bias baseline" \
        "  pcqr_cub200_arm_vpt_{v,qv}.yaml                        prompt context" \
        "  pcqr_cub200_arm_vptsp_{v,qv}.yaml                      proxy context"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            CONFIG="configs/pcqr_$2.yaml"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --epochs)
            SETS+=("n_epochs=$2")
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --source-path)
            SETS+=("source_path=$2")
            shift 2
            ;;
        --set)
            SETS+=("$2")
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --dry-run)
            DRY=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            printf 'error: unknown argument: %s\n\n' "$1" >&2
            print_usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    printf 'error: config not found: %s\n\navailable:\n' "$CONFIG" >&2
    ls -1 configs/*.yaml >&2
    exit 2
fi

if [[ -z "$NAME" ]]; then
    NAME="$(basename "$CONFIG" .yaml)"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    printf 'error: interpreter not found: %s\n' "$PYTHON_BIN" >&2
    printf 'hint: pass --python /path/to/python or set PYTHON_BIN.\n' >&2
    exit 2
}

LAUNCH_DIR="$PROJECT_ROOT/results/_launch"
mkdir -p "$LAUNCH_DIR"
EFFECTIVE_CONFIG="$LAUNCH_DIR/$NAME.yaml"

RENDER_ARGS=(--config "$CONFIG" --out "$EFFECTIVE_CONFIG" --set "name=$NAME")
for item in ${SETS[@]+"${SETS[@]}"}; do
    RENDER_ARGS+=(--set "$item")
done

printf 'project : %s\n' "$PROJECT_ROOT"
printf 'config  : %s\n' "$CONFIG"
printf 'run name: %s\n' "$NAME"
printf 'gpu     : %s\n' "$GPU"

if [[ "$DRY" -eq 1 ]]; then
    RENDER_ARGS+=(--require-paths --print)
    "$PYTHON_BIN" prepare_config.py "${RENDER_ARGS[@]}"
    printf '\ndry run complete; nothing was trained.\n'
    exit 0
fi

"$PYTHON_BIN" prepare_config.py "${RENDER_ARGS[@]}"
printf '\nstarting training...\n\n'

exec "$PYTHON_BIN" -u train.py --config "$EFFECTIVE_CONFIG" --gpu "$GPU"
