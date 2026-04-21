# AGENTS.md

## Project Identity
This repository contains a commercial-grade offline smart glasses system for blind and visually impaired users, designed for Raspberry Pi 4.

The system must prioritize:
- safety
- low latency
- modularity
- maintainability
- offline operation
- real-world robustness
- Raspberry Pi 4 optimization

This is not a toy demo project.

---

## Platform Constraints
- Hardware: Raspberry Pi 4 + Camera Module 2
- OS: Raspberry Pi OS Bookworm
- Python: 3.11
- Offline-only after setup
- Edge-optimized inference
- Real-time or near-real-time behavior preferred

---

## Product Scope
Core modules:
- object detection
- obstacle warning
- face detection
- face recognition
- emotion detection
- OCR
- offline TTS audio feedback
- decision engine
- state machine
- diagnostics and logging

---

## Architecture Rules
1. Keep the code modular.
2. Do not create a giant monolithic main file.
3. Separate camera, inference, decision-making, and audio handling.
4. Use config-driven settings wherever possible.
5. Prefer reusable service classes and manager classes.
6. Keep inference pipelines event-driven and efficient.
7. Use frame skipping, cooldowns, and ROI-based processing.
8. Add clear extension points for future features.
9. Write production-style code, not notebook-style code.
10. Every module should have a single clear responsibility.

---

## Performance Rules
1. Optimize specifically for Raspberry Pi 4.
2. Avoid running all models on every frame.
3. Face recognition must only run after valid face detection and quality checks.
4. Emotion detection must run less frequently than face detection.
5. OCR must be on-demand or event-triggered where possible.
6. Audio must be queued and deduplicated.
7. Critical alerts must interrupt low-priority messages.
8. Use lightweight models and backends suitable for Pi.

---

## Safety and UX Rules
1. Danger alerts have highest priority.
2. Avoid repeated speech spam.
3. Use cooldown timers for repeated events.
4. Unknown or low-confidence results should be handled conservatively.
5. Never force a confident output when confidence is low.
6. Favor reliability over flashy behavior.
7. System behavior must degrade gracefully under load.

---

## State Machine Expectations
Supported modes:
- NORMAL
- DANGER
- FACE
- READING
- QUIET
- LOW_POWER

Each mode should affect:
- active modules
- frame interval behavior
- cooldown values
- speech behavior
- CPU-saving strategy

---

## Face Recognition Rules
- Support local enrollment and local storage of embeddings
- Support add/remove/update person entries
- Apply quality checks before recognition:
  - min face size
  - blur threshold
  - brightness threshold
  - stable detection
- Use configurable similarity thresholds
- Use cooldowns for repeated identity announcements
- Unknown faces should be handled explicitly

---

## Object Detection Rules
Focus on navigation-relevant classes:
- person
- bicycle
- motorcycle
- car
- bus
- truck
- chair
- table
- dog
- cat
- traffic light
- stop sign

Use heuristics for:
- closeness
- center danger zone
- left/right guidance
- severity scoring

---

## OCR Rules
- OCR should be preprocessed for better edge-device accuracy
- Cache recent OCR outputs
- Avoid repeating same text too often
- Treat OCR as important but not danger-critical unless configured otherwise

---

## Audio Rules
Priority levels:
- P1 danger
- P2 identity/OCR
- P3 object/navigation
- P4 low-priority context

Rules:
- P1 interrupts all
- P2 can replace P3/P4
- P3/P4 should be deduplicated and rate-limited
- stale messages should expire

---

## Code Generation Rules for Codex
When generating or modifying code:
1. Generate complete runnable code.
2. Prefer robust implementations over shortcuts.
3. Avoid placeholders unless absolutely necessary.
4. If a real model file is required, create integration-ready loader code and document expected paths.
5. Maintain import consistency across files.
6. Keep path handling correct for Raspberry Pi Linux environment.
7. Use clear type hints where helpful.
8. Add concise, useful docstrings.
9. Preserve offline-first design.
10. Keep codebase maintainable and commercial in style.

---

## Repository Structure Expectations
Expected high-level directories:
- app/
- core/
- services/
- models/
- audio/
- vision/
- ocr/
- face/
- config/
- scripts/
- tests/
- docs/

Exact structure may evolve, but architecture must stay modular.

---

## Documentation Rules
- Keep README clear and professional
- Include setup, install, run, calibration, and troubleshooting steps
- Document model file expectations
- Document performance tuning knobs
- Document future extension points

---

## Future Feature Readiness
Design the system so these can be added later:
- GPS guidance
- voice commands
- SOS
- currency recognition
- medication reading
- scene description
- haptic feedback
- battery monitoring
- multilingual support including Sinhala

---

## Main Goal
Every implementation choice should move the repository toward a robust, offline, real-world smart glasses product for visually impaired users on Raspberry Pi 4.