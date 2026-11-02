"""Retain polling-only provider usage as a separate, auditable token category."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.campaign / "response-usage.csv").open() as stream:
        usage = list(csv.DictReader(stream))
    with args.audit.open() as stream:
        audit = list(csv.DictReader(stream))
    key = lambda row: (row["trial_id"], row["response_id"])
    audited = {key(row): row for row in audit}
    if len(audited) != len(audit) or len({key(row) for row in usage}) != len(usage):
        raise ValueError("Response identifiers must be unique")
    if set(audited) != {key(row) for row in usage}:
        raise ValueError("Polling audit must cover exactly the collected responses")
    totals = defaultdict(Counter)
    for row in usage:
        record = audited[key(row)]
        if record["polling_only"] not in {"True", "False"}:
            raise ValueError("polling_only must be True or False")
        count = int(row["total_tokens"])
        if count != int(record["total_tokens"]) or row["arm"] != record["arm"]:
            raise ValueError("Polling audit differs from provider usage")
        if count != int(row["input_tokens"]) + int(row["output_tokens"]):
            raise ValueError("Total tokens must equal input plus output")
        category = "polling_only" if record["polling_only"] == "True" else "other"
        totals[row["arm"]][category + "_tokens"] += count
        totals[row["arm"]][category + "_responses"] += 1
        totals[row["arm"]]["total_tokens"] += count
    fields = [
        "arm",
        "total_tokens",
        "polling_only_tokens",
        "other_tokens",
        "polling_only_responses",
        "other_responses",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, counts in sorted(totals.items()):
            writer.writerow({"arm": arm, **{name: counts[name] for name in fields[1:]}})


if __name__ == "__main__":
    main()
