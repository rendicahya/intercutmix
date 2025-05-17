import os
from datetime import datetime, timedelta

import click


def parse_timestamp(line):
    try:
        return datetime.strptime(line[:19], "%Y/%m/%d %H:%M:%S")
    except ValueError:
        return None


def get_start_time(lines, search):
    for line in lines:
        if search in line:
            timestamp = parse_timestamp(line)
            if timestamp:
                return timestamp
    return None


def get_end_time(lines):
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        timestamp = parse_timestamp(line)
        if timestamp:
            return timestamp
    return None


def find_log_file(folder):
    """Recursively find the first .log file in the folder."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".log"):
                return os.path.join(root, file)
    return None


@click.command()
def calculate_average_duration():
    base_folder = click.prompt(
        "Please enter the path to the base folder (level 1)", type=click.Path(exists=True, file_okay=False)
    )

    search = "mmengine - INFO"
    durations = {"test": [], "train": []}

    level2_folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder)
                      if os.path.isdir(os.path.join(base_folder, d))]

    for level2_folder in level2_folders:
        for mode in ["test", "train"]:
            level3_folder = os.path.join(level2_folder, mode)
            if not os.path.isdir(level3_folder):
                click.echo(f"Skipping missing folder: {level3_folder}")
                continue

            log_file = find_log_file(level3_folder)
            if not log_file:
                click.echo(f"No log file found in {level3_folder}")
                continue

            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start_time = get_start_time(lines, search)
            end_time = get_end_time(lines)

            if not start_time or not end_time:
                click.echo(f"Could not find valid timestamps in {log_file}")
                continue

            duration = end_time - start_time
            durations[mode].append(duration)
            click.echo(f"{mode.upper()} - {os.path.basename(level2_folder)}: {duration}")

    # Calculate averages
    for mode in ["test", "train"]:
        if durations[mode]:
            total = sum((d.total_seconds() for d in durations[mode]))
            avg_seconds = total / len(durations[mode])
            avg_duration = timedelta(seconds=avg_seconds)
            click.echo(f"\nAverage duration for {mode}: {avg_duration} ({avg_seconds:.0f} seconds)")
        else:
            click.echo(f"\nNo valid durations found for {mode}.")


if __name__ == "__main__":
    calculate_average_duration()
