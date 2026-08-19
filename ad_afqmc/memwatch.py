import os
import sys
import time

root_pid = int(sys.argv[1])
interval = 0.5

peak_pss_kb = 0
peak_rss_kb = 0


def descendants(root):
    """Return root PID + all descendant process PIDs. Threads are ignored."""
    result = {root}
    changed = True

    while changed:
        changed = False

        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue

            pid = int(name)

            try:
                with open(f"/proc/{pid}/stat") as f:
                    stat = f.read()

                # Field 4 is PPID.
                # Handle process names containing spaces/parentheses safely.
                rest = stat[stat.rfind(")") + 2:].split()
                ppid = int(rest[1])

                if ppid in result and pid not in result:
                    result.add(pid)
                    changed = True

            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass

    return result


def memory_kb(pid):
    """Return (RSS, PSS) from smaps_rollup."""
    rss = 0
    pss = 0

    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss = int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass

    return rss, pss


while os.path.exists(f"/proc/{root_pid}"):
    pids = descendants(root_pid)

    total_rss = 0
    total_pss = 0

    for pid in pids:
        rss, pss = memory_kb(pid)
        total_rss += rss
        total_pss += pss

    peak_rss_kb = max(peak_rss_kb, total_rss)
    peak_pss_kb = max(peak_pss_kb, total_pss)

    print(
        f"\rprocesses={len(pids):2d}  "
        f"PSS={total_pss / 1024**2:7.2f} GiB  "
        f"RSS={total_rss / 1024**2:7.2f} GiB  "
        f"peak PSS={peak_pss_kb / 1024**2:7.2f} GiB",
        end="",
        flush=True,
    )

    time.sleep(interval)


print()
print(f"Peak PSS: {peak_pss_kb / 1024**2:.2f} GiB")
print(f"Peak RSS: {peak_rss_kb / 1024**2:.2f} GiB")