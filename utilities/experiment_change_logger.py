"""Records configuration changes, best Recall@1 and run status."""

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import re
import signal
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

import options as parameter_library


BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
STATE_FILENAME = ".experiment_log_state.json"
TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})(?:_(\d+))?$")

STATUS_NAMES = {
    "RUNNING": "running",
    "COMPLETED": "completed",
    "INTERRUPTED": "stopped early",
    "FAILED": "failed",
}

# These parameters only affect paths, devices, plotting or terminal
# output; they are not experimental variables.
IGNORED_CONFIG_KEYS = {
    "config", "save_path", "gpu", "name", "kernels", "world_size",
    "time", "show_trainable_pars", "progress_update_interval",
    "save_results", "save_step",
    "storage_metrics", "store_improvements", "heatmap_max_images",
    "tsne_num_classes", "tsne_samples_per_class", "disable_plots",
    "evaluation_metrics", "evaltypes",
}

CHANGE_PRIORITY = (
    "source_path", "pretrained_model", "arch", "loss", "seed",
    "n_epochs", "batch_size", "samples_per_class", "augmentation",
    "model_input_size", "embed_dim", "optim", "lr", "lr_proxy",
    "decay", "scheduler", "warmup", "mix_precision",
    "eval_before_training", "eval_start", "eval_interval",
    "fast_evaluation",
    "subset_experiment_enabled", "subset_experiment_protocol",
    "quick_train_class_count", "quick_test_class_count",
    "full_fine_tune", "train_head", "visual_prompt", "semantic_proxy",
    "all_bias_bitfit", "vit_rc_bitfit", "rc_bitfit_start_layer", "adapter",
    "num_prompt", "prompt_depth",
    "num_p_prompt", "p_prompt_depth", "proxy_net_depth",
    "proxy_net_head", "proxy_data_range", "semantic_mix_type",
    "proxy_mix_bias", "proxy_mix_keep_ratio", "proxy_bias_ratio",
)

DISPLAY_NAMES = {
    "source_path": "data source",
    "pretrained_model": "pretrained model",
    "arch": "architecture",
    "loss": "objective",
    "seed": "random seed",
    "n_epochs": "epochs",
    "batch_size": "batch size",
    "samples_per_class": "samples per class",
    "augmentation": "augmentation",
    "model_input_size": "input resolution",
    "embed_dim": "embedding dim",
    "optim": "optimiser",
    "lr": "base learning rate",
    "lr_proxy": "proxy learning rate",
    "decay": "weight decay",
    "scheduler": "lr scheduler",
    "warmup": "warmup epochs",
    "eval_before_training": "evaluate before training",
    "eval_start": "first evaluated epoch",
    "eval_interval": "evaluation interval",
    "fast_evaluation": "fast evaluation",
    "semantic_mix_type": "proxy aggregation",
}

_ACTIVE_LOGGER = None


def _now():
    return datetime.now(BEIJING_TIMEZONE)


def _now_text():
    return _now().isoformat(timespec="seconds")


def _process_start_ticks(pid):
    try:
        stat_text = Path("/proc/{}/stat".format(int(pid))).read_text(
            encoding="utf-8")
        fields = stat_text[stat_text.rfind(")") + 2:].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _process_is_live(pid, expected_ticks):
    actual_ticks = _process_start_ticks(pid)
    return (actual_ticks is not None and expected_ticks is not None and
            int(expected_ticks) == actual_ticks)


def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as input_file:
            value = yaml.safe_load(input_file)
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _safe_component(value, limit=30):
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_.")
    return (value or "none")[:limit]


def _format_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with open(temporary, "w", encoding="utf-8") as output_file:
        output_file.write(text)
    os.replace(str(temporary), str(path))


def _atomic_json(path, payload):
    _atomic_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2,
                         default=str) + "\n")


def _timestamp_rank(run_dir):
    match = TIMESTAMP_PATTERN.fullmatch(Path(run_dir).name)
    if not match:
        return None
    timestamp = datetime.strptime(
        match.group(1), "%Y_%m_%d_%H_%M_%S").replace(
            tzinfo=BEIJING_TIMEZONE)
    return timestamp.timestamp(), int(match.group(2) or 0)


def _file_hash(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as input_file:
            while True:
                block = input_file.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()[:12]
    except OSError:
        return "missing"


def _code_fingerprints(project_root, config):
    candidates = {
        "main.py": project_root / "main.py",
        "params.py": project_root / "params.py",
    }
    loss_name = str(config.get("loss", "")).strip()
    arch_name = str(config.get("arch", "")).strip()
    if loss_name:
        candidates["criterion"] = project_root / "criteria" / (
            loss_name + ".py")
        if "ratefit" in loss_name:
            candidates["rate_reduction"] = (
                project_root / "criteria" / "rate_reduction.py")
    if arch_name:
        candidates["architecture"] = project_root / "architectures" / (
            arch_name + ".py")
    return {name: _file_hash(path) for name, path in candidates.items()}


def _relevant_new_key(key, config):
    if key in CHANGE_PRIORITY:
        return True
    loss_name = str(config.get("loss", ""))
    active_prefixes = ["loss_oproxy_"]
    if "nir" in loss_name:
        active_prefixes.append("loss_nir_")
    if "simdino" in loss_name:
        active_prefixes.extend(("loss_simdino_", "loss_simpeft_"))
    if "repulsion" in loss_name:
        active_prefixes.append("loss_proxy_repulsion_")
    if "ratefit" in loss_name:
        active_prefixes.append("loss_ratefit_")
    return any(key.startswith(prefix) for prefix in active_prefixes)


def _config_changes(reference, current):
    if not reference:
        return []
    changes = []
    for key in sorted(set(reference) | set(current)):
        if key in IGNORED_CONFIG_KEYS:
            continue
        if key not in reference and not _relevant_new_key(key, current):
            continue
        if key not in current:
            continue
        old_value = reference.get(key)
        new_value = current.get(key)
        if old_value != new_value:
            changes.append({"key": key, "old": old_value, "new": new_value})
    priority = {name: index for index, name in enumerate(CHANGE_PRIORITY)}
    changes.sort(key=lambda item: (
        priority.get(item["key"], len(priority)), item["key"]))
    return changes


def _code_default_config():
    """Build the params.py code defaults before any CLI or YAML override."""
    parser = argparse.ArgumentParser(add_help=False)
    parser = parameter_library.basic_training_parameters(parser)
    parser = parameter_library.scale_optimizing_parameters(parser)
    parser = parameter_library.batch_creation_parameters(parser)
    parser = parameter_library.batchmining_specific_parameters(parser)
    parser = parameter_library.loss_specific_parameters(parser)
    parser = parameter_library.nir_parameters(parser)
    parser = parameter_library.simdino_parameters(parser)
    return vars(parser.parse_args([]))


def _canonical_default_config(project_root, current):
    """Code defaults merged with the canonical ``*_basic.yaml`` baseline."""
    defaults = _code_default_config()
    dataset = _safe_component(current.get("dataset", ""), limit=40)
    baseline_path = project_root / "config" / "{}_basic.yaml".format(dataset)
    baseline = _load_yaml(baseline_path)
    if baseline:
        for key, wrapped in baseline.items():
            if isinstance(wrapped, dict) and "value" in wrapped:
                defaults[key] = wrapped["value"]
            else:
                defaults[key] = wrapped
        label = "params.py + {}".format(baseline_path.name)
    else:
        label = "params.py argparse defaults"
    return defaults, label


def _change_filename(changes, code_changes, config, comparison="previous"):
    prefix = "experiment_log__vs_{}".format(comparison)
    if not changes and not code_changes:
        return "{}__none.txt".format(prefix)
    tokens = []
    for change in changes[:4]:
        key = change["key"]
        value = change["new"]
        if key == "source_path":
            value = Path(str(value)).name
        elif isinstance(value, bool):
            value = "on" if value else "off"
        tokens.append("{}-{}".format(
            _safe_component(key, 24), _safe_component(value, 30)))
    if code_changes and len(tokens) < 4:
        tokens.append("code-changed")
    remaining = len(changes) + (1 if code_changes else 0) - len(tokens)
    if remaining > 0:
        tokens.append("plus{}".format(remaining))
    body = "__".join(tokens)[:170]
    return "{}__{}.txt".format(prefix, body)


def _find_reference(dataset_root, current_run):
    current_rank = _timestamp_rank(current_run)
    if current_rank is None:
        return None, {}, {}
    candidates = []
    for config_path in dataset_root.rglob("effective_parameters.yaml"):
        run_dir = config_path.parent
        if run_dir.resolve() == current_run.resolve():
            continue
        if any(part in {"Logs", ".cleanup_trash", "cleanup_logs"}
               for part in config_path.parts):
            continue
        rank = _timestamp_rank(run_dir)
        if rank is not None and rank < current_rank:
            candidates.append((rank, run_dir, config_path))
    if not candidates:
        return None, {}, {}
    _, run_dir, config_path = max(candidates, key=lambda item: item[0])
    reference_state = {}
    state_path = run_dir / STATE_FILENAME
    if state_path.is_file():
        try:
            reference_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            reference_state = {}
    return run_dir, _load_yaml(config_path), reference_state


def _duration_text(state):
    try:
        start = datetime.fromisoformat(state["started_at_beijing"])
        end_text = state.get("ended_at_beijing") or state.get(
            "updated_at_beijing")
        end = datetime.fromisoformat(end_text)
        seconds = max(0, int((end - start).total_seconds()))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    except (KeyError, TypeError, ValueError):
        return "unknown"


def _result_text(state):
    best = state.get("best_recall_at_1")
    if best is None:
        return "no test result yet"
    return "{:.6f} @ epoch {}".format(
        float(best), state.get("best_recall_at_1_epoch"))


def _render_run_log(state):
    status = state.get("status", "RUNNING")
    status_text = STATUS_NAMES.get(status, status)
    lines = [
        "Experiment configuration and iteration log",
        "=" * 72,
        "status        : {} ({})".format(status_text, status),
        "started       : {}".format(state.get("started_at_beijing", "")),
        "last update   : {}".format(
            state.get("ended_at_beijing") or state.get("updated_at_beijing", "")),
        "duration      : {}".format(_duration_text(state)),
        "experiment    : {}".format(state.get("experiment_name", "")),
        "run directory : {}".format(state.get("run_dir", "")),
        "baseline      : {}".format(
            state.get("default_baseline", "params.py argparse defaults")),
        "previous run  : {}".format(
            state.get("reference_run") or "none (first comparable experiment)"),
        "full config   : {}".format(state.get("effective_config_path", "")),
        "",
        "Results",
        "-" * 72,
        "Best Recall@1 : {}".format(_result_text(state)),
        "last test epoch : {}".format(
            state.get("latest_test_epoch")
            if state.get("latest_test_epoch") is not None else "not tested yet"),
        "planned epochs  : {}".format(state.get("n_epochs", "unknown")),
        "",
        "Key identifiers",
        "-" * 72,
        "dataset       : {}".format(state.get("dataset", "")),
        "data source   : {}".format(state.get("source_path", "")),
        "model         : {} / {}".format(
            state.get("arch", ""), state.get("pretrained_model", "")),
        "loss          : {}".format(state.get("loss", "")),
        "seed          : {}".format(state.get("seed", "")),
        "",
        "Differences from the canonical default configuration",
        "-" * 72,
    ]
    default_changes = state.get("default_changes", [])
    if default_changes:
        for change in default_changes:
            label = DISPLAY_NAMES.get(change["key"], change["key"])
            lines.append("- {} ({}): {}  ->  {}".format(
                label, change["key"], _format_value(change.get("old")),
                _format_value(change.get("new"))))
    elif state.get("version", 1) >= 2:
        lines.append("- matches the canonical default configuration.")
    else:
        lines.append("- legacy v1 logs did not store the default comparison; "
                         "see the next section.")

    lines.extend([
        "", "Iteration changes relative to the most recent previous run", "-" * 72,
    ])
    previous_changes = state.get("previous_changes", state.get("changes", []))
    if previous_changes:
        for change in previous_changes:
            label = DISPLAY_NAMES.get(change["key"], change["key"])
            lines.append("- {} ({}): {}  ->  {}".format(
                label, change["key"], _format_value(change.get("old")),
                _format_value(change.get("new"))))
    else:
        lines.append("- no iteration change, or no comparable previous run.")

    lines.extend(["", "Code changes", "-" * 72])
    code_changes = state.get("code_changes", [])
    if code_changes:
        for change in code_changes:
            lines.append("- {}: {}  ->  {}".format(
                change["file"], change.get("old", "unknown"),
                change.get("new", "unknown")))
    else:
        lines.append("- no code change detected, or the previous run has no "
                         "code fingerprint.")
    lines.append("- current short fingerprint: {}".format(
        ", ".join("{}={}".format(key, value) for key, value in
                  sorted(state.get("code_fingerprints", {}).items()))))

    if state.get("termination_reason"):
        lines.extend([
            "", "Closing notes", "-" * 72,
            "- {}".format(state["termination_reason"]),
        ])
    lines.extend([
        "", "Notes",
        "-" * 72,
        "- The canonical baseline is params.py plus the dataset *_basic.yaml.",
        "- Iteration changes are compared with the most recent previous run on "
            "the same dataset.",
        "- Full parameters are stored in effective_parameters.yaml.",
        "- RUNNING/COMPLETED/INTERRUPTED/FAILED denote running, completed, "
            "stopped early and failed.",
        "",
    ])
    return "\n".join(lines)


def _daily_entry(state, dataset_root):
    status = STATUS_NAMES.get(state.get("status"), state.get("status"))
    best = _result_text(state)
    default_changes = []
    for change in state.get("default_changes", [])[:6]:
        default_changes.append("{}={}".format(
            change["key"], _format_value(change.get("new"))))
    default_extra = len(state.get("default_changes", [])) - len(default_changes)
    if default_extra > 0:
        default_changes.append("{} more".format(default_extra))
    previous_changes = []
    for change in state.get("previous_changes", state.get("changes", []))[:6]:
        previous_changes.append("{}:{}→{}".format(
            change["key"], _format_value(change.get("old")),
            _format_value(change.get("new"))))
    extra = len(state.get("previous_changes", state.get("changes", []))) - len(previous_changes)
    if extra > 0:
        previous_changes.append("{} more".format(extra))
    if state.get("code_changes"):
        previous_changes.append("code changed")
    try:
        relative_run = str(Path(state["run_dir"]).resolve().relative_to(
            dataset_root.resolve()))
    except (KeyError, OSError, ValueError):
        relative_run = state.get("run_dir", "")
    return [
        "[{}] {} | R@1 {} | seed={} | {} / {}".format(
            state.get("started_at_beijing", "")[11:19], status, best,
            state.get("seed", ""), state.get("pretrained_model", ""),
            state.get("loss", "")),
        "  non-default: {}".format(
            "; ".join(default_changes) if default_changes else "none"),
        "  vs previous: {}".format(
            "; ".join(previous_changes) if previous_changes
                                     else "none or no previous baseline"),
        "  directory: {}".format(relative_run),
    ]


@contextmanager
def _dataset_lock(logs_dir):
    logs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = logs_dir / ".experiment_logs.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read_states(dataset_root):
    states = []
    for state_path in dataset_root.rglob(STATE_FILENAME):
        if "Logs" in state_path.parts:
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["_state_path"] = str(state_path)
            states.append(state)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return states


def _refresh_daily_logs_locked(dataset_root, logs_dir):
    states = _read_states(dataset_root)
    now_text = _now_text()
    for state in states:
        if state.get("status") == "RUNNING" and not _process_is_live(
                state.get("pid", -1), state.get("process_start_ticks")):
            state["status"] = "INTERRUPTED"
            state["ended_at_beijing"] = now_text
            state["updated_at_beijing"] = now_text
            state["termination_reason"] = (
                "the process ended without a completion marker; recorded as stopped "
                "early.")
            state_path = Path(state.pop("_state_path"))
            _atomic_json(state_path, state)
            _atomic_write(
                Path(state["run_dir"]) / state["log_filename"],
                _render_run_log(state))
        else:
            state.pop("_state_path", None)

    grouped = {}
    for state in states:
        date_key = state.get("started_at_beijing", "")[:10].replace("-", "_")
        if date_key:
            grouped.setdefault(date_key, []).append(state)
    for date_key, date_states in grouped.items():
        date_states.sort(key=lambda state: (
            state.get("started_at_beijing", ""), state.get("run_dir", "")))
        counts = {name: sum(state.get("status") == name for state in date_states)
                  for name in STATUS_NAMES}
        lines = [
            "{} experiment log".format(date_key),
            "=" * 96,
            "updated: {}".format(now_text),
            "total: {} | running: {} | completed: {} | stopped: {} | failed: {}".format(
                len(date_states), counts["RUNNING"], counts["COMPLETED"],
                counts["INTERRUPTED"], counts["FAILED"]),
            "",
        ]
        for state in date_states:
            lines.extend(_daily_entry(state, dataset_root))
            lines.append("")
        _atomic_write(
            logs_dir / "{}_experiments.txt".format(date_key),
            "\n".join(lines))


class ExperimentChangeLogger:
    """Maintain one run log and refresh the per-dataset daily log safely."""

    def __init__(self, run_dir, effective_config_path):
        global _ACTIVE_LOGGER
        self.owner_pid = os.getpid()
        self.run_dir = Path(run_dir).resolve()
        self.config_path = Path(effective_config_path).resolve()
        self.config = _load_yaml(self.config_path)
        self.dataset_root = self.run_dir.parent.parent
        self.logs_dir = self.dataset_root / "Logs"
        self.state_path = self.run_dir / STATE_FILENAME
        self.closed = False

        with _dataset_lock(self.logs_dir):
            reference_run, reference_config, reference_state = _find_reference(
                self.dataset_root, self.run_dir)
            previous_changes = _config_changes(reference_config, self.config)
            project_root = Path(__file__).resolve().parent.parent
            default_config, default_baseline = _canonical_default_config(
                project_root, self.config)
            default_changes = _config_changes(default_config, self.config)
            fingerprints = _code_fingerprints(project_root, self.config)
            previous_fingerprints = reference_state.get(
                "code_fingerprints", {})
            code_changes = []
            if previous_fingerprints:
                for name in sorted(set(previous_fingerprints) | set(fingerprints)):
                    old_value = previous_fingerprints.get(name, "missing")
                    new_value = fingerprints.get(name, "missing")
                    if old_value != new_value:
                        code_changes.append({
                            "file": name, "old": old_value, "new": new_value})
            # The file name uses the stable canonical baseline, so reordering previous
            # runs never renames the same configuration. Iteration differences are
            # still stored in full in the body and in the state file.
            log_filename = _change_filename(
                default_changes, code_changes, self.config,
                comparison="defaults")
            now_text = _now_text()
            self.state = {
                "version": 2,
                "status": "RUNNING",
                "pid": self.owner_pid,
                "process_start_ticks": _process_start_ticks(self.owner_pid),
                "started_at_beijing": now_text,
                "updated_at_beijing": now_text,
                "ended_at_beijing": None,
                "experiment_name": str(self.config.get("name", "")),
                "dataset": str(self.config.get("dataset", "")),
                "source_path": str(self.config.get("source_path", "")),
                "arch": str(self.config.get("arch", "")),
                "pretrained_model": str(
                    self.config.get("pretrained_model", "")),
                "loss": str(self.config.get("loss", "")),
                "seed": self.config.get("seed"),
                "n_epochs": self.config.get("n_epochs"),
                "run_dir": str(self.run_dir),
                "effective_config_path": str(self.config_path),
                "reference_run": str(reference_run) if reference_run else "",
                # "changes" is kept for compatibility; it always means "vs previous run".
                "changes": previous_changes,
                "default_baseline": default_baseline,
                "default_changes": default_changes,
                "previous_changes": previous_changes,
                "code_fingerprints": fingerprints,
                "code_changes": code_changes,
                "best_recall_at_1": None,
                "best_recall_at_1_epoch": None,
                "latest_test_epoch": None,
                "termination_reason": "",
                "log_filename": log_filename,
            }
            self._write_own_files_locked()
            _refresh_daily_logs_locked(self.dataset_root, self.logs_dir)

        _ACTIVE_LOGGER = self
        atexit.register(self._atexit)
        self._register_signal_handlers()

    @property
    def log_path(self):
        return self.run_dir / self.state["log_filename"]

    def _write_own_files_locked(self):
        _atomic_json(self.state_path, self.state)
        _atomic_write(self.log_path, _render_run_log(self.state))

    def _persist(self):
        if os.getpid() != self.owner_pid:
            return
        self.state["updated_at_beijing"] = _now_text()
        with _dataset_lock(self.logs_dir):
            self._write_own_files_locked()
            _refresh_daily_logs_locked(self.dataset_root, self.logs_dir)

    def update_metrics(self):
        if self.closed or os.getpid() != self.owner_pid:
            return
        metrics_path = self.run_dir / "records" / "best_metrics.json"
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            recall = payload.get("metrics", {}).get("Recall@1", {})
            if recall.get("best") is not None:
                self.state["best_recall_at_1"] = float(recall["best"])
                self.state["best_recall_at_1_epoch"] = int(recall["epoch"])
            if payload.get("latest_test_epoch") is not None:
                self.state["latest_test_epoch"] = int(
                    payload["latest_test_epoch"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        self._persist()

    def mark_completed(self):
        if self.closed or os.getpid() != self.owner_pid:
            return
        self.update_metrics()
        self.state["status"] = "COMPLETED"
        self.state["ended_at_beijing"] = _now_text()
        self.state["termination_reason"] = "training loop finished normally."
        self.closed = True
        self._persist_final()

    def mark_interrupted(self, reason="training stopped by the user or the system."):
        if self.closed or os.getpid() != self.owner_pid:
            return
        self.update_metrics()
        self.state["status"] = "INTERRUPTED"
        self.state["ended_at_beijing"] = _now_text()
        self.state["termination_reason"] = str(reason)
        self.closed = True
        self._persist_final()

    def mark_failed(self, error):
        if self.closed or os.getpid() != self.owner_pid:
            return
        self.update_metrics()
        self.state["status"] = "FAILED"
        self.state["ended_at_beijing"] = _now_text()
        self.state["termination_reason"] = "{}: {}".format(
            type(error).__name__, error)
        self.closed = True
        self._persist_final()

    def _persist_final(self):
        self.state["updated_at_beijing"] = _now_text()
        with _dataset_lock(self.logs_dir):
            self._write_own_files_locked()
            _refresh_daily_logs_locked(self.dataset_root, self.logs_dir)

    def _atexit(self):
        if not self.closed and os.getpid() == self.owner_pid:
            self.mark_interrupted("the process exited before the training loop "
                                  "completed.")

    def _register_signal_handlers(self):
        if os.getpid() != self.owner_pid:
            return
        for signal_number in (signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(signal_number, self._handle_signal)
            except (OSError, RuntimeError, ValueError):
                pass

    def _handle_signal(self, signal_number, _frame):
        self.mark_interrupted("received system signal {}.".format(signal_number))
        signal.signal(signal_number, signal.SIG_DFL)
        os.kill(os.getpid(), signal_number)


def get_active_experiment_logger():
    return _ACTIVE_LOGGER


def mark_active_failed(error):
    logger = get_active_experiment_logger()
    if logger is not None:
        try:
            logger.mark_failed(error)
        except Exception as log_error:
            print("Warning: failed to record experiment failure: {}".format(
                log_error))


def mark_active_interrupted(reason):
    logger = get_active_experiment_logger()
    if logger is not None:
        try:
            logger.mark_interrupted(reason)
        except Exception as log_error:
            print("Warning: failed to record experiment interruption: {}".format(
                log_error))
