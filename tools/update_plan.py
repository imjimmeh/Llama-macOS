import re

with open('docs/plans/2026-08-21-routing-lookahead-pipeline.md', 'r') as f:
    content = f.read()

old = """## Phase 5D: Learned Low-Rank Routing Predictor

**Objective:** Train tiny external model to predict future expert routes.

**Duration:** 7-10 days  
**Risk:** Medium-High (requires training infrastructure, but model is tiny)"""

new = """## Phase 5D: Learned Low-Rank Routing Predictor

**Objective:** Train tiny external model to predict future expert routes.

**Status:** Implementation complete but **not integrated** into execution path.

**Limitation:** The learned predictor requires hidden state data from the forward pass, but the `mul_mat_id` execution path only has access to the expert weights tensor (`input`), not the hidden state that feeds the router. Integration would require passing hidden state through the scheduler/graph execution infrastructure, which is a deeper architectural change.

**Current deliverables:**
- API declarations in `ggml-backend-expert-cache.h`
- C++ implementation in `ggml-backend-expert-cache.cpp` (model loading, inference)
- Python training script in `tools/train_routing_predictor.py`
- **Not integrated** into `ggml-backend.cpp` execution path

**Path forward:** Test heuristic predictor (Phase 5C) first. If it shows meaningful speedup, the learned predictor can be integrated later with additional infrastructure changes.

**Duration:** 7-10 days (original estimate)  
**Risk:** Medium-High"""

content = content.replace(old, new)

with open('docs/plans/2026-08-21-routing-lookahead-pipeline.md', 'w') as f:
    f.write(content)

print('Updated Phase 5D status in plan document')
