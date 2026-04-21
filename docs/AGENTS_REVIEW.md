# AGENTS.md Review

## Verdict

The current `AGENTS.md` is a strong production brief for an offline Raspberry Pi 4 assistive smart glasses project. It gives clear safety, performance, UX, and modularity constraints. The guidance is consistent with a commercial-grade edge system and does not conflict with the generated configuration structure.

## Strengths

- Safety and low latency are correctly treated as primary product constraints.
- The state machine modes are explicit and map cleanly to runtime performance profiles.
- Face recognition is gated by face detection and quality checks, which prevents wasteful and unsafe identity guesses.
- OCR is scoped as event-triggered, which is appropriate for Raspberry Pi 4.
- Audio priority rules are clear enough to drive queue, interruption, deduplication, and stale-message behavior.
- Future capabilities are named without forcing the first implementation to overbuild.

## Production Gaps Addressed By This Pass

- Concrete model path definitions were added in `config/settings.py`.
- Thresholds and cooldowns were centralized into immutable settings objects and made tunable through `config/runtime.toml`.
- Raspberry Pi 4 optimization presets were formalized as `pi4_conservative`, `pi4_balanced`, and `pi4_responsive`.
- Mode-specific performance profiles were defined for `NORMAL`, `DANGER`, `FACE`, `READING`, `QUIET`, and `LOW_POWER`, with camera FPS capped by the selected Pi profile.
- Environment variables now have a documented structure in `.env.example` and `docs/ENVIRONMENT.md`.
- Disabled schedules use `None` instead of `0`, avoiding divide-by-zero risks in future schedulers.
- Audio priority comparison is explicit: lower numeric priority is more urgent, and profiles expose helper methods for allow checks.
- Model assets are tiered by criticality so safety-critical startup can be separated from feature degradation.

## Recommended Future Additions

- Fill `models/manifest.local.toml` with real model licensing, provenance, hashes, and schemas during setup.
- Add privacy rules for face embeddings, retention, export, and deletion.
- Add explicit safety test expectations for danger alerts, quiet mode, stale audio, and thermal throttling.
- Add release criteria for field testing on actual Raspberry Pi 4 hardware.
