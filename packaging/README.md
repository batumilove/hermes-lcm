# Runtime distribution

The installable subdirectory is `packages/hermes-lcm`. Manifest v2 declares all five dependency bounds. The native entrypoint is `__init__.py:register`.

Build with `python3 packaging/build_runtime.py NEW_DESTINATION`. The deterministic builder refuses existing destinations, symlinks, traversal and changed source hashes. The allowlist preserves all 49 runtime modules; the generated manifest and provenance receipt bring the delivery to 51 files. Tests and hostile synthetic fixtures remain in the development source, not the runtime artifact.

## Portable offline check

Provision a compatible self-contained Python prefix and its declared dependencies separately. Run on Linux with bubblewrap and namespace permission:

```sh
python3 packaging/run_isolated.py \
  --host-source /path/to/compatible/hermes-agent \
  --python-runtime /path/to/self-contained/python \
  --dependencies /path/to/provisioned/site-packages \
  --output /path/to/new/private-results
```

Use `--sudo` only when namespace creation requires noninteractive sudo. In this mode namespace setup retains actual UID 0 (no user namespace), drops all capabilities, and owns only the newly created private HOME and temporary directory; it does not change supplier permissions. Unprivileged mode requires suppliers readable by the mapped UID. The runner checks that Python resolves within its supplied prefix and that its stdlib exists, constructs its own network-disabled namespace, uses a private HOME and read-only suppliers, and stops on a failed check. It does not read historical receipts, install dependencies, enable a production plugin, or invoke lifecycle commands. Results include complete scanner findings and subprocess exit status. The root filesystem is read-only and home/run are hidden; this is a trusted-source integration fixture, not a hostile-code security sandbox.

The behavioral probe exercises registration, tools, session-end checkpointing, module import provenance and token counting against the delivered tree. It is not a supported-installer rehearsal or paid embedding/model-asset qualification. Namespace failures are failures, not skips. Run development tests with declared pytest in a separate disposable HOME; do not use operator state. Runtime scanner findings must remain visible; synthetic security fixtures must not be deleted to obtain a clean scan.

Source publication is not installation, runtime activation, independent review, or CI acceptance. Operational receipts remain outside this repository.
