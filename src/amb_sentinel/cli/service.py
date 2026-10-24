"""`sentinel service install|uninstall|start|stop|restart|status`.

Generates a systemd unit (Linux), launchd plist (macOS), or Task Scheduler
XML (Windows). `--dry-run` prints the generated file and exits without
touching the system, letting CI exercise every branch on every OS.

Scope:
  --user    (default) — current user.
              Linux:   ~/.config/systemd/user/ (survives logout via linger)
              macOS:   ~/Library/LaunchAgents/ (runs while user is logged in)
              Windows: current-user Task Scheduler at logon
  --system  Linux only — /etc/systemd/system/. Requires root.
              macOS/Windows `--system` is unsupported (use sudo/elevated shell
              with the platform-native tool instead).

Home directory (base for config.yaml, .env, logs, data, cache):
  --home <path>       explicit override
  $AMB_HOME           environment variable
  current working dir (default)
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Sequence

from amb_sentinel import paths


# ── helpers ──────────────────────────────────────────────────────────────────
def _interp() -> str:
    """The interpreter that installed the package (pipx and venv both work)."""
    return sys.executable


def _run(cmd: Sequence[str]) -> int:
    """Run a subprocess, print it, return its exit code. Never shell=True."""
    print(f"  $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        print(f"  ! {cmd[0]}: not found on PATH")
        return 127


def _run_quiet(cmd: Sequence[str]) -> int:
    try:
        return subprocess.run(cmd, check=False, capture_output=True).returncode
    except FileNotFoundError:
        return 127


def _runtime_user(system: bool) -> str:
    """Who the service runs as."""
    if not system:
        return getpass.getuser()
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _write(path: Path, content: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mode is not None:
        try:
            os.chmod(path, mode)
        except OSError:
            pass

def _resolve_home(args) -> Path:
    """Return the base directory for runtime state (config, logs,
    checkpoints, backups).

    Resolution order:
      1. --home <path>           (explicit override)
      2. $AMB_HOME               (env export reaching the CLI)
      3. Path.cwd()              (default — the directory you ran the
                                  command from)

    Relative paths are resolved against the current working directory.
    ~ and ${VAR} are expanded.
    """
    explicit = getattr(args, "home", None)
    if explicit:
        return Path(
            os.path.expandvars(os.path.expanduser(str(explicit)))
        ).resolve()

    env_home = (os.environ.get("AMB_HOME") or "").strip()
    if env_home:
        return Path(
            os.path.expandvars(os.path.expanduser(env_home))
        ).resolve()

    return Path.cwd().resolve()


def _read_active_home(system: bool) -> Path | None:
    """Read the pointer file the installer wrote. Used by status to show
    the right log path even when the CLI is invoked from elsewhere."""
    if system:
        pointer = Path("/etc/amb-sentinel/paths.json")
    else:
        pointer = Path.home() / ".config" / "amb-sentinel" / "paths.json"
    if not pointer.exists():
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
        home = data.get("amb_home")
        if home:
            return Path(home)
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None

def _canonical_config_path(home: Path) -> Path:
    """Where the running service reads config.yaml from.

    The home directory is the base for all runtime state; config.yaml
    lives directly inside it, not in a subdirectory.
    """
    return home / "config.yaml"


# ── Linux / systemd ──────────────────────────────────────────────────────────
def _systemd_unit(
    *,
    home: Path,
    system: bool,
    user: str | None,
    memory_max: str | None,
    cpu_quota: str | None,
) -> str:
    cfg = _canonical_config_path(home)
    env = home / ".env"
    user_line = f"User={user}\n" if system else ""
    install_target = "multi-user.target" if system else "default.target"

    limit_lines = ""
    if memory_max:
        limit_lines += f"MemoryMax={memory_max}\n"
    if cpu_quota:
        limit_lines += f"CPUQuota={cpu_quota}\n"

    return f"""\
[Unit]
Description=AMB-Sentinel autonomous AI agent
Documentation=https://github.com/michealbamidele147-cmd/AMB-APP
After=network-online.target postgresql.service redis.service local-llm-cloud.service
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
{user_line}WorkingDirectory={home}
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=AMB_HOME={home}
Environment=AMB_CONFIG={cfg}
EnvironmentFile=-{env}
ExecStart={_interp()} -m amb_sentinel run
Restart=on-failure
RestartSec=10
RestartPreventExitStatus=78
LimitNOFILE=65536
{limit_lines}
[Install]
WantedBy={install_target}
"""

def _install_linux(*, system: bool, dry: bool, args) -> int:
    home = _resolve_home(args)
    user = _runtime_user(system) if system else None

    unit = _systemd_unit(
        home=home,
        system=system,
        user=user,
        memory_max=getattr(args, "memory_max", None),
        cpu_quota=getattr(args, "cpu_quota", None),
    )

    if dry:
        print(unit)
        return 0

    if system and os.geteuid() != 0:
        print("  ! --system requires root (run with sudo)")
        return 1

    if not home.exists():
        print(f"  ! home directory does not exist: {home}")
        return 1

    cfg = _canonical_config_path(home)
    if not cfg.exists():
        src = paths.find_config()
        if src:
            shutil.copy(src, cfg)
            try:
                if system and user:
                    shutil.chown(cfg, user=user, group=user)
            except LookupError:
                pass
            os.chmod(cfg, 0o600)
            print(f"  seeded {cfg} from {src}")
        else:
            print(f"  ! no config.yaml in {home} and no source to seed from.")
            print("    Put config.yaml in that directory, or pass --home <dir>.")
            return 1

    for sub in ("logs", "data", "cache",
                "data/execution_backups", "data/agent_checkpoints"):
        d = home / sub
        d.mkdir(parents=True, exist_ok=True)
        if system and user:
            try:
                shutil.chown(d, user=user, group=user)
            except LookupError:
                pass

    if system:
        _i = _interp()
        if _i.startswith("/home/") or _i.startswith("/Users/"):
            print(f"  ! warning: interpreter {_i} lives under a user home;")
            print(f"    the '{user}' service user may not be able to run it.")
            print("    Prefer `sudo pipx install amb-sentinel` or a system-wide pip install.")

        pointer = Path("/etc/amb-sentinel/paths.json")
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(
            json.dumps({"amb_home": str(home)}) + "\n",
            encoding="utf-8",
        )
        os.chmod(pointer, 0o644)
        print(f"  wrote {pointer}")

        target = Path("/etc/systemd/system/amb-sentinel.service")
        scope_args: list[str] = []
    else:
        pointer = Path.home() / ".config" / "amb-sentinel" / "paths.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(
            json.dumps({"amb_home": str(home)}) + "\n",
            encoding="utf-8",
        )
        os.chmod(pointer, 0o600)
        print(f"  wrote {pointer}")

        target = Path.home() / ".config" / "systemd" / "user" / "amb-sentinel.service"
        scope_args = ["--user"]

    _write(target, unit)
    print(f"  wrote {target}")

    _run(["systemctl"] + scope_args + ["daemon-reload"])
    _run(["systemctl"] + scope_args + ["enable", "amb-sentinel"])
    if not system:
        _run(["loginctl", "enable-linger", getpass.getuser()])
    print(f"  next: systemctl {' '.join(scope_args)} start amb-sentinel")
    return 0

# ── macOS / launchd ──────────────────────────────────────────────────────────
def _launchd_plist(home: Path) -> str:
    log_dir = home / "logs"
    cfg = _canonical_config_path(home)
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>io.amb.sentinel</string>
  <key>ProgramArguments</key>
  <array>
    <string>{_interp()}</string>
    <string>-m</string>
    <string>amb_sentinel</string>
    <string>run</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{home}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>StandardOutPath</key>
  <string>{log_dir}/stdout.log</string>
  <key>StandardErrorPath</key>
  <string>{log_dir}/stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
    <key>AMB_HOME</key>
    <string>{home}</string>
    <key>AMB_CONFIG</key>
    <string>{cfg}</string>
  </dict>
</dict>
</plist>
"""

def _install_macos(*, dry: bool, args) -> int:
    home = _resolve_home(args)
    plist = _launchd_plist(home)
    if dry:
        print(plist)
        return 0

    if not home.exists():
        print(f"  ! home directory does not exist: {home}")
        return 1

    cfg = _canonical_config_path(home)
    if not cfg.exists():
        src = paths.find_config()
        if src:
            shutil.copy(src, cfg)
            os.chmod(cfg, 0o600)
            print(f"  seeded {cfg} from {src}")
        else:
            print(f"  ! no config.yaml in {home} and no source to seed from.")
            return 1

    for sub in ("logs", "data", "cache",
                "data/execution_backups", "data/agent_checkpoints"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    pointer = Path.home() / ".config" / "amb-sentinel" / "paths.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"amb_home": str(home)}) + "\n", encoding="utf-8")
    print(f"  wrote {pointer}")

    target = Path.home() / "Library" / "LaunchAgents" / "io.amb.sentinel.plist"
    _write(target, plist)
    print(f"  wrote {target}")

    uid = os.getuid()
    label = f"gui/{uid}/io.amb.sentinel"
    _run(["launchctl", "bootstrap", f"gui/{uid}", str(target)])
    _run(["launchctl", "kickstart", label])
    return 0

# ── Windows / Task Scheduler ─────────────────────────────────────────────────
def _scheduler_xml(*, at_boot: bool, home: Path) -> str:
    trigger = (
        "<BootTrigger><Enabled>true</Enabled></BootTrigger>"
        if at_boot
        else "<LogonTrigger><Enabled>true</Enabled></LogonTrigger>"
    )
    cfg = _canonical_config_path(home)
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AMB-Sentinel autonomous AI agent</Description>
  </RegistrationInfo>
  <Triggers>{trigger}</Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RestartOnFailure>
      <Interval>PT30S</Interval>
      <Count>999</Count>
    </RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{_interp()}</Command>
      <Arguments>-m amb_sentinel run</Arguments>
    </Exec>
  </Actions>
</Task>
"""

def _install_windows(*, dry: bool, at_boot: bool, args) -> int:
    home = _resolve_home(args)
    xml = _scheduler_xml(at_boot=at_boot, home=home)
    if dry:
        print(xml)
        return 0

    if not home.exists():
        print(f"  ! home directory does not exist: {home}")
        return 1

    for sub in ("logs", "data", "cache",
                "data/execution_backups", "data/agent_checkpoints"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    xml_file = Path(os.environ.get("TEMP", str(Path.home()))) / "amb-sentinel-task.xml"
    _write(xml_file, xml, mode=0o600)
    print(f"  wrote {xml_file}")

    rc = _run(["schtasks", "/Create", "/TN", "AMB-Sentinel", "/XML", str(xml_file), "/F"])
    if rc == 0:
        _run(["schtasks", "/Run", "/TN", "AMB-Sentinel"])
    else:
        print("  ! schtasks failed. If you used --at-boot, run this from an elevated shell.")
    return rc

# ── install dispatcher ───────────────────────────────────────────────────────
def _cmd_install(args) -> int:
    sysname = platform.system()
    dry = getattr(args, "dry_run", False)
    system = _resolve_system_scope(args, action="install")

    if sysname == "Linux":
        if not system:
            _require_user_session("install")
        return _install_linux(system=system, dry=dry, args=args)
    if sysname == "Darwin":
        if system:
            print("  ! --system is not supported on macOS; using user scope (LaunchAgent)")
        return _install_macos(dry=dry, args=args)
    if sysname == "Windows":
        if system:
            print("  ! --system on Windows means an elevated shell; using user scope")
        return _install_windows(dry=dry, at_boot=getattr(args, "at_boot", False), args=args)
    print(f"  ! unsupported platform: {sysname}")
    return 1

# ── uninstall ────────────────────────────────────────────────────────────────
def _cmd_uninstall(args) -> int:
    sysname = platform.system()
    system = _resolve_system_scope(args, action="uninstall")

    if sysname == "Linux":
        if not system:
            _require_user_session("uninstall")
        scope = [] if system else ["--user"]
        _run(["systemctl"] + scope + ["disable", "--now", "amb-sentinel"])

        unit_path = (
            Path("/etc/systemd/system/amb-sentinel.service") if system
            else Path.home() / ".config" / "systemd" / "user" / "amb-sentinel.service"
        )
        if unit_path.exists():
            unit_path.unlink()
            print(f"  removed {unit_path}")

        _run(["systemctl"] + scope + ["daemon-reload"])

        # Remove pointer file (both scopes) — this is metadata about the
        # location of state, not the state itself. Uninstalling must not
        # delete the home directory; user data stays.
        for ptr in (
            Path("/etc/amb-sentinel/paths.json"),
            Path.home() / ".config" / "amb-sentinel" / "paths.json",
        ):
            if ptr.exists():
                ptr.unlink()
                print(f"  removed {ptr}")

        print(f"  note: state directory was NOT removed. Delete it manually if desired.")
        return 0

    if sysname == "Darwin":
        uid = os.getuid()
        plist = Path.home() / "Library" / "LaunchAgents" / "io.amb.sentinel.plist"
        _run(["launchctl", "bootout", f"gui/{uid}", str(plist)])
        if plist.exists():
            plist.unlink()
            print(f"  removed {plist}")
        return 0

    if sysname == "Windows":
        _run(["schtasks", "/End", "/TN", "AMB-Sentinel"])
        _run(["schtasks", "/Delete", "/TN", "AMB-Sentinel", "/F"])
        return 0

    print(f"  ! unsupported platform: {sysname}")
    return 1

# ── lifecycle (start / stop / restart) ───────────────────────────────────────
def _make_lifecycle(action: str):
    def _run_action(args) -> int:
        sysname = platform.system()
        system = _resolve_system_scope(args, action=action)

        if sysname == "Linux":
            if not system:
                _require_user_session(action)
            scope = [] if system else ["--user"]
            return _run(["systemctl"] + scope + [action, "amb-sentinel"])

        if sysname == "Darwin":
            uid = os.getuid()
            label = f"gui/{uid}/io.amb.sentinel"
            if action == "start":
                return _run(["launchctl", "kickstart", label])
            if action == "stop":
                return _run(["launchctl", "kill", "SIGTERM", label])
            if action == "restart":
                return _run(["launchctl", "kickstart", "-k", label])
            return 1

        if sysname == "Windows":
            flag = {"start": "/Run", "stop": "/End", "restart": "/End"}.get(action)
            if flag is None:
                return 1
            return _run(["schtasks", flag, "/TN", "AMB-Sentinel"])

        print(f"  ! unsupported platform: {sysname}")
        return 1
    return _run_action


# ── status ───────────────────────────────────────────────────────────────────
def _cmd_status(args) -> int:
    sysname = platform.system()
    system = _resolve_system_scope(args, action="status")
    rc = 0

    if sysname == "Linux":
        if not system:
            _require_user_session("status")
        scope = [] if system else ["--user"]
        rc = _run(["systemctl"] + scope + ["status", "--no-pager", "amb-sentinel"])
    elif sysname == "Darwin":
        uid = os.getuid()
        rc = _run(["launchctl", "print", f"gui/{uid}/io.amb.sentinel"])
    elif sysname == "Windows":
        rc = _run(["schtasks", "/Query", "/TN", "AMB-Sentinel", "/V", "/FO", "LIST"])
    else:
        print(f"  ! unsupported platform: {sysname}")
        return 1

    # Prefer the pointer file the installer wrote, so status reports the
    # correct log location even when run from a different directory.
    active_home = _read_active_home(system)
    if active_home is not None:
        log_dir = active_home / "logs"
    else:
        log_dir = paths.log_dir()

    print()
    print(f"  log directory: {log_dir}")
    ai_log = log_dir / "ai_agent.log"
    if ai_log.exists():
        try:
            tail = ai_log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
            for ln in tail:
                print(f"  | {ln}")
        except OSError as e:
            print(f"  ! could not read {ai_log}: {e}")
    else:
        print(f"  ({ai_log.name} not written yet)")
    return rc

def _add_scope_flags(p: argparse.ArgumentParser) -> None:
    """--system and --user are mutually exclusive. If neither is passed,
    scope is resolved at runtime (root on Linux → system, else user)."""
    g = p.add_mutually_exclusive_group()
    g.add_argument("--system", action="store_true",
                   help="Linux only: system-wide scope (requires root)")
    g.add_argument("--user", action="store_true",
                   help="Per-user scope (default for non-root)")

def _resolve_system_scope(args, *, action: str) -> bool:
    """Decide between system and user scope.

    Priority:
      1. --system explicitly  → True
      2. --user explicitly    → False
      3. Neither, root+Linux  → True, with a printed note
      4. Otherwise            → False
    """
    if getattr(args, "system", False):
        return True
    if getattr(args, "user", False):
        return False
    if (platform.system() == "Linux"
            and hasattr(os, "geteuid")
            and os.geteuid() == 0):
        print(f"  note: running as root without --user; using --system for {action}")
        return True
    return False


def _has_user_session_bus() -> bool:
    """True iff this shell can talk to `systemctl --user`.

    Requires XDG_RUNTIME_DIR (systemd user manager socket directory)
    and a reachable session bus. The bus may be in
    DBUS_SESSION_BUS_ADDRESS or at $XDG_RUNTIME_DIR/bus.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR", "")
    if not xdg:
        return False
    if os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return True
    return (Path(xdg) / "bus").exists()


def _require_user_session(action: str) -> None:
    """Exit with a clear, actionable message if `systemctl --user` cannot work."""
    if _has_user_session_bus():
        return
    print("  ! No systemd user session available in this shell.")
    print(f"    XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR')!r}")
    print(f"    DBUS_SESSION_BUS_ADDRESS={os.environ.get('DBUS_SESSION_BUS_ADDRESS')!r}")
    print()
    print(f"  {action} needs a running user manager. Options:")
    print("    1) Use system scope (recommended when you're root):")
    print(f"         sudo sentinel service {action} --system")
    print("    2) Get a user session first:")
    print("         loginctl enable-linger <your-user>")
    print("         (then log out and log back in)")
    raise SystemExit(1)

# ── parser wiring ────────────────────────────────────────────────────────────
def add_subparser(sub) -> None:
    p = sub.add_parser("service", help="Install or manage the OS service")
    sp = p.add_subparsers(dest="svc_cmd")

    inst = sp.add_parser("install", help="Install the service for this OS")
    _add_scope_flags(inst)
    inst.add_argument(
        "--home", default=None,
        help="Base directory for runtime state (default: current directory). "
             "config.yaml is read from <home>/config.yaml, logs go to "
             "<home>/logs/, etc. Also honoured: $AMB_HOME when --home is "
             "not passed."
    )
    inst.add_argument("--dry-run", action="store_true",
                      help="Print the generated unit/plist/XML and exit")
    inst.add_argument("--memory-max", default=None,
                      help="Linux only: MemoryMax= override (e.g. 8G)")
    inst.add_argument("--cpu-quota", default=None,
                      help="Linux only: CPUQuota= override (e.g. 80%%)")
    inst.add_argument("--at-boot", action="store_true",
                      help="Windows only: trigger at boot (requires elevated shell)")
    inst.set_defaults(func=_cmd_install)

    uninst = sp.add_parser("uninstall", help="Remove the installed service")
    _add_scope_flags(uninst)
    uninst.set_defaults(func=_cmd_uninstall)

    for action in ("start", "stop", "restart"):
        cmd = sp.add_parser(action, help=f"{action.title()} the service")
        _add_scope_flags(cmd)
        cmd.set_defaults(func=_make_lifecycle(action))

    st = sp.add_parser("status", help="Report service state and recent log lines")
    _add_scope_flags(st)
    st.add_argument(
        "--home", default=None,
        help="Override the log-directory source shown in status output "
             "(default: read from the installer's pointer file)."
    )
    st.set_defaults(func=_cmd_status)

    p.set_defaults(func=lambda _a: p.print_help() or 0)
