import os
import sys
import time
import csv

root_pid = int(sys.argv[1])
interval = 0.5

peak_pss_kb = 0
peak_rss_kb = 0
process_peaks = {}

start_time = time.monotonic()
lines_printed = 0

csv_path = f"{root_pid}.csv"


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
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return "?"


with open(csv_path, "w", newline="", buffering=1) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "elapsed_s",
        "pid",
        "name",
        "pss_gib",
        "rss_gib",
        "peak_pss_gib",
    ])

    while os.path.exists(f"/proc/{root_pid}"):

        elapsed = time.monotonic() - start_time

        pids = descendants(root_pid)

        total_rss = 0
        total_pss = 0
        current = []

        for pid in sorted(pids):

            rss, pss = memory_kb(pid)

            if rss == 0 and pss == 0:
                continue

            total_rss += rss
            total_pss += pss

            process_peaks[pid] = max(
                process_peaks.get(pid, 0),
                pss,
            )

            name = process_name(pid)

            current.append(
                (pid, name, rss, pss, process_peaks[pid])
            )

        peak_rss_kb = max(peak_rss_kb, total_rss)
        peak_pss_kb = max(peak_pss_kb, total_pss)

        # ----------------------------------------
        # Write this sample to CSV
        # ----------------------------------------

        # Total row
        writer.writerow([
            f"{elapsed:.3f}",
            root_pid,
            "TOTAL",
            f"{total_pss / 1024**2:.6f}",
            f"{total_rss / 1024**2:.6f}",
            f"{peak_pss_kb / 1024**2:.6f}",
        ])

        # Individual process rows
        for pid, name, rss, pss, peak in current:
            writer.writerow([
                f"{elapsed:.3f}",
                pid,
                name,
                f"{pss / 1024**2:.6f}",
                f"{rss / 1024**2:.6f}",
                f"{peak / 1024**2:.6f}",
            ])

        # ----------------------------------------
        # Terminal display
        # ----------------------------------------

        elapsed_int = int(elapsed)
        hours, rem = divmod(elapsed_int, 3600)
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

        if lines_printed:
            print(f"\033[{lines_printed}F", end="")

        for line in lines:
            print(f"\033[2K{line}")

        for _ in range(lines_printed - len(lines)):
            print("\033[2K")

        lines_printed = len(lines)

        time.sleep(interval)


elapsed = time.monotonic() - start_time

print("\nProcess finished.")
print(f"Elapsed time: {elapsed:.1f} s")
print(f"Peak total PSS: {peak_pss_kb / 1024**2:.2f} GiB")
print(f"Peak total RSS: {peak_rss_kb / 1024**2:.2f} GiB")
print(f"Data saved to: {csv_path}")

print("\nPer-process peak PSS:")
for pid, peak in sorted(
    process_peaks.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {pid:8d}: {peak / 1024**2:.2f} GiB")
