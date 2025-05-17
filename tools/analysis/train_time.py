from datetime import datetime

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


@click.command()
def calculate_duration():
    log_file = click.prompt(
        "Please enter the path to the log file", type=click.Path(exists=True)
    )

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    search = "mmengine - INFO"
    start_time = get_start_time(lines, search)

    if not start_time:
        click.echo(f"Could not find a line with '{search}'.")
        return

    end_time = get_end_time(lines)

    if not end_time:
        click.echo("Could not find a valid timestamp in the last non-empty line.")
        return

    duration = end_time - start_time
    click.echo(f"Start time: {start_time}")
    click.echo(f"End time:   {end_time}")
    click.echo(f"Duration:   {duration} ({duration.total_seconds():.0f} seconds)")


if __name__ == "__main__":
    calculate_duration()
