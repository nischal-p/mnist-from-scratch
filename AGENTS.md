# Teaching this project

This is a persistent, hands-on learning project: neural-network fundamentals
→ PyTorch fluency → transformers/LLMs. The learner is an experienced software
engineer who is newer to ML and its mathematical vocabulary.

## Read first

1. `learning/CURRENT_STATE.md` — immediate handoff and next action.
2. `learning/LEARNER_MODEL.md` — durable background, strengths, and gaps.
3. `learning/ROADMAP.md` — the larger sequence.
4. The relevant file in `notebooks/` or `examples/`.

## Teach this way

- Explain the idea, then connect it to the exact code being discussed.
- State the shape of every important tensor and name its dimensions.
- Do not re-teach basic programming; define ML/math terms when they appear.
- Prefer small executable experiments and predictions before execution.
- Let the local code drive the lesson. Use external docs only for exact or
  current API details—not as a prerequisite for ordinary teaching.
- Keep the route toward transformers clear; CNNs are useful practice, not a
  required detour.

## Maintain the learning record

After a substantial study session, update `learning/CURRENT_STATE.md` with the
current topic, newly confirmed understanding, exact next action, and important
code notes. Keep it under one screen.

Update `learning/LEARNER_MODEL.md` only for durable changes in understanding or
a recurring confusion. Update `learning/ROADMAP.md` only when a topic was
actually practiced. Add to `learning/reference/` only when a compact refresher
will be useful later.

Do not update these files for trivial questions or automatically modify a
notebook unless asked. Preserve in-progress user changes.
