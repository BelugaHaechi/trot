import os
import sys
import time

root_pid = int(sys.argv[1])
interval = 0.5

peak_pss_kb = 0
peak_rss_kb = 0

# Per-PID peak PSS
process_peaks = {}


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

                # Everything after the final ")" starts with:
                # state, ppid, ...
                rest = stat[stat.rfind(")") + 2:].split()
                ppid = int(rest[1])

                if ppid in result and pid not in result:
                    result.add(pid)
                    changed = True

            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass

    return result


def memory_kb(pid):
    """Return (RSS, PSS) in KiB from smaps_rollup."""
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


def process_name(pid):
    """Return command name for PID."""
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return "?"

start_time = time.monotonic()
lines_printed = 0
while os.path.exists(f"/proc/{root_pid}"):

    pids = descendants(root_pid)

    total_rss = 0
    total_pss = 0
    current = []
    for pid in sorted(pids):

        rss, pss = memory_kb(pid)

        # Process may have disappeared between discovery and measurement
        if rss == 0 and pss == 0:
            continue

        total_rss += rss
        total_pss += pss

        # Update this PID's peak
        process_peaks[pid] = max(
            process_peaks.get(pid, 0),
            pss,
        )

        current.append(
            (pid, process_name(pid), rss, pss, process_peaks[pid])
        )

    peak_rss_kb = max(peak_rss_kb, total_rss)
    peak_pss_kb = max(peak_pss_kb, total_pss)

    # time elapsed
    elapsed = time.monotonic() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    lines = []

    lines.append(
        f"TOTAL"
        f"    elapsed={elapsed_str}"
        f"    PSS={total_pss / 1024**2:8.2f} GiB"
        f"    RSS={total_rss / 1024**2:8.2f} GiB"
        f"    peak PSS={peak_pss_kb / 1024**2:8.2f} GiB"
    )

    lines.append("")
    lines.append(
        f"{'PID':>8} "
        f"{'NAME':<16} "
        f"{'PSS GiB':>10} "
        f"{'RSS GiB':>10} "
        f"{'PEAK PSS':>10}"
    )
    lines.append("-" * 60)

    for pid, name, rss, pss, peak in current:
        lines.append(
            f"{pid:8d} "
            f"{name:<16.16} "
            f"{pss / 1024**2:10.2f} "
            f"{rss / 1024**2:10.2f} "
            f"{peak / 1024**2:10.2f}"
        )

    # Move back to beginning of our previous output
    if lines_printed:
        print(f"\033[{lines_printed}F", end="")

    # Print each line, clearing that line first
    for line in lines:
        print(f"\033[2K{line}")

    # If the previous display was taller, erase leftover lines
    for _ in range(lines_printed - len(lines)):
        print("\033[2K")

    lines_printed = len(lines)
    time.sleep(interval)

# total time elapsed
elapsed = time.monotonic() - start_time
hours, rem = divmod(int(elapsed), 3600)
minutes, seconds = divmod(rem, 60)

print("\nProcess finished.")
print(f"Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")
print(f"Peak total PSS: {peak_pss_kb / 1024**2:.2f} GiB")
print(f"Peak total RSS: {peak_rss_kb / 1024**2:.2f} GiB")

print("\nPer-process peak PSS:")
for pid, peak in sorted(
    process_peaks.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {pid:8d}: {peak / 1024**2:.2f} GiB")