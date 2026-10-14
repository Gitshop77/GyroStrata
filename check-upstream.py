"""Read upstream commit IDs without modifying the installed, pinned sources."""

import json, pathlib, subprocess

pins = json.loads(
    (pathlib.Path(__file__).resolve().parent / "pins.json").read_text(
        encoding="utf-8-sig"
    )
)
for name in ["strata", "llama"]:
    pin = pins[name]
    remote = subprocess.check_output(
        ["git", "ls-remote", pin["url"], "HEAD"], text=True, timeout=60
    ).split()[0]
    print(
        name,
        "tested pin:",
        pin["revision"],
        "upstream HEAD:",
        remote,
        "(unchanged)"
        if remote == pin["revision"]
        else "(newer/different; validation required)",
    )
print("No source, configuration, driver or model file was changed.")
