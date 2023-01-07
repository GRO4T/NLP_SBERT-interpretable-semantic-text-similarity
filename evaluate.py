import os
from subprocess import check_output

from lib.utils import TYPES_MAP
from lib.params import DATA_DIR, DATASET

fields_sep = " // "


def preds_to_wa(wa_content: str, preds_lines):
    wa_lines = wa_content.splitlines()

    idx = 0
    result = []

    for line in wa_lines:
        line_res = line

        if "<==>" in line:
            fields = line.split(fields_sep)
            preds_fields = preds_lines[idx].split()

            fields[1] = preds_fields[1]
            fields[2] = preds_fields[2]

            line_res = fields_sep.join(fields)
            idx += 1

        result.append(line_res)

    return "\n".join(result)


def flatten(t):
    return [item for sublist in t for item in sublist]


def evaluate(predictions):
    print(type(predictions))
    print(predictions[0])
    # Generate predictions
    types_inv_map = {v: k for k, v in TYPES_MAP.items()}

    types = list(
        map(lambda t: types_inv_map[t], flatten([t.tolist() for t, s in predictions]))
    )
    scores = flatten([s.tolist() for t, s in predictions])

    predictions = [
        f"{index}\t{item[0]} {item[1]}\n"
        for index, item in enumerate(zip(types, scores))
    ]

    # Create wa file with predictions
    wa_file = os.path.join(DATA_DIR, f"STSint.testinput.{DATASET}.wa")
    wa_output_file = os.path.join(
        DATA_DIR, f"STSint.testinput.{DATASET}-predictions.wa"
    )

    with open(wa_file) as file:
        wa_test = file.read()

    wa_predictions = preds_to_wa(wa_test, predictions)

    with open(wa_output_file, "w") as file:
        file.write(wa_predictions)

    # Run Perl eval scripts
    cmds = [
        f"perl evalF1_penalty.pl {wa_file} {wa_output_file}",
        f"perl evalF1_no_penalty.pl {wa_file} {wa_output_file}",
    ]

    for cmd in cmds:
        print(f"Executing {cmd}")
        print(check_output(cmd.split(), cwd="./").decode())
