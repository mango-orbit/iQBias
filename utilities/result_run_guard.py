"""Maintain a verifiable liveness marker for a running experiment."""

import atexit
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone


BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _process_start_ticks(pid):
    """Read the Linux process start clock so a reused PID cannot be mistaken
    for the original process."""
    try:
        stat_text = Path("/proc/{}/stat".format(int(pid))).read_text(
            encoding="utf-8")
        fields_after_name = stat_text[stat_text.rfind(")") + 2:].split()
        return int(fields_after_name[19])
    except (OSError, ValueError, IndexError):
        return None


class ActiveRunGuard:
    """Write the liveness lock into the run directory and clear it on normal
    or abnormal exit."""

    FILENAME = ".active_run.json"

    def __init__(self, run_dir, experiment_name=""):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / self.FILENAME
        self.temporary_path = self.run_dir / ".active_run.tmp.json"
        self.pid = os.getpid()
        self.start_ticks = _process_start_ticks(self.pid)
        self.closed = False

        payload = {
            "pid": self.pid,
            "process_start_ticks": self.start_ticks,
            "experiment_name": str(experiment_name),
            "run_dir": str(self.run_dir),
            "created_at_beijing": datetime.now(BEIJING_TIMEZONE).isoformat(),
        }
        with open(self.temporary_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        os.replace(str(self.temporary_path), str(self.path))
        atexit.register(self.close)

    def close(self):
        """Remove only the lock owned by this process instance, never a later
        run's marker."""
        if self.closed:
            return
        self.closed = True
        try:
            with open(self.path, "r", encoding="utf-8") as input_file:
                payload = json.load(input_file)
            if (int(payload.get("pid", -1)) == self.pid and
                    payload.get("process_start_ticks") == self.start_ticks):
                self.path.unlink()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        try:
            self.temporary_path.unlink()
        except OSError:
            pass
