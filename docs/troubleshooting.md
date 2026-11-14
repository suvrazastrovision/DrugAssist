# Troubleshooting

[README](../README.md) | [Setup](setup.md) | [Usage](usage.md) | [Tests](testing.md) | [Troubleshooting](troubleshooting.md)

| Problem | Try this |
|---|---|
| `uv` / `git` not recognized | Reopen the terminal; check the [tool installation instructions](setup.md#1-install-the-tools). |
| Installation: `Access is denied` | Stop servers/notebooks, pause folder syncing and retry. Persistent incomplete metadata may require recreating `.venv`. |
| Prepared download fails | Check the [data release](https://github.com/suvrazastrovision/DrugAssist/releases/tag/data-v0.1.0) and [platform requirements](setup.md#3-set-up-and-launch); explicitly choose a source build if needed. |
| Port already in use | Stop the earlier app. Windows launcher alternatives: `-ApiPort 8001 -UiPort 8502`. |
| Search/answer fails | Check API key and billing/access. Launcher logs: `results/private/startup/`. |

Stop services before changing dependencies/data. Use a separate checkout for a fresh configuration.
Prepared downloads verify checksums, preserve credentials/accounts and never automatically fall back to a paid build.

## Langfuse traces are missing

- Install the `observability` extra using the [observability guide](observability.md).
- Set `DRUGASSIST_OBSERVABILITY_ENABLED=true` and configure both Langfuse keys and the project's regional base URL.
- Restart the API, submit a search or answer request, then refresh Langfuse after a few seconds.
- Check the selected project, time range and trace filters. Health checks and login requests do not create traces.

Missing question, answer and passage text is expected: this integration exports metadata only.
Telemetry failures do not prevent the app from answering requests.
