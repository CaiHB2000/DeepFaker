This file tracks high-priority edits to make before the ARR submission deadline.
Items are ordered from highest impact / lowest cost to more ambitious tasks.

Each item should be checked off (and commit message noted) once completed.

---

## A. Scope, positioning, and wording (no new experiments)

1. **Tighten title and scope**
   - [ ] Rename the paper to make the setting explicit, e.g.:
     - `Reliability-Aware Gated Distillation from Multi-Branch Teachers for Multimodal Rumor Detection`
   - [ ] In the Introduction (Section 1), add a short paragraph early on that clearly states:
     - We assume a fixed multimodal teacher with explicit text/image/fusion branches.
     - We focus on supervised multimodal rumor datasets with explicit event IDs (Weibo, WeFEND).
     - We do *not* claim a general KD framework for arbitrary models or unlabeled data.
   - [ ] Move or duplicate the current “Scope and Assumptions” paragraph from the end of the Method section to an earlier location (Intro or start of Method), so reviewers see it before details.

2. **Downgrade contribution wording from “new framework” to “systematic instantiation”**
   - [ ] In the Contributions list, rephrase:
     - Emphasize that DMPD instantiates and systematically studies agreement- and event-aware distillation in this specific multimodal rumor setting.
     - Explicitly acknowledge that key components (agreement-based gating, curriculum under noise, evidential heads) are inspired by prior work; the contribution is in how they are combined and evaluated here.

3. **Soften main-results claims**
   - [ ] In the Experiments “Main Results” subsection:
     - [ ] Describe Weibo gains as “small but consistent” (~+0.5pt Macro-F1 over SAFE/EANN) and calibration as “comparable to the fusion teacher”.
     - [ ] Describe WeFEND gains as “~0.8–1pt Macro-F1 and ~0.01 lower ECE than SAFE/teacher”.
     - [ ] Describe MiRAGeNews as a stress test where the student recovers ~2pt Macro-F1 under generator shift with similar or slightly better ECE, without calling the gains “substantial”.

4. **Make Fakeddit a clearly articulated hard case**
   - [ ] In Limitations, add an explicit bullet that:
     - Summarizes the strict time-split Fakeddit results (all content baselines low; DMPD does not outperform teacher).
     - States that this regime is treated as a hard case where DMPD does not help weak teachers.
   - [ ] Ensure Appendix Fakeddit paragraph and Limitations use consistent language.

5. **Weaken “calibration-first” as a headline contribution**
   - [ ] In Contributions, merge/soften the “Calibration-first objective” point to say:
     - The objective avoids worsening ECE and sometimes improves it (not a primary standalone contribution).
   - [ ] In the main text, avoid framing calibration as the central novelty; instead present calibration improvements as a secondary benefit of the gating + loss design.

---

## B. Use existing experiments to strengthen comparisons

6. **Integrate Simple-Conf-KD as an explicit baseline**
   - [x] In Experiments/Appendix, add a short paragraph that:
     - Defines the Simple-Conf-KD baseline (student distills whenever teacher confidence exceeds a threshold, no cross-view or event logic).
     - Reports its performance on Weibo/WeFEND (using existing `weibo_simple_conf_kd/*` and `wefend_simple_conf_kd/*` summaries).
     - Clearly states that Simple-Conf-KD underperforms DMPD in Macro-F1 and ECE, supporting the value of cross-view + event-aware gating.

7. **Clarify the MiRAGeNews ID/OOD story**
   - [x] In Experiments, expand the MiRAGeNews paragraph to:
     - Distinguish ID (test1) vs OOD (avg test2–5) splits.
     - Summarize teacher vs DMPD on both (numbers already available from existing eval JSONs).
   - [ ] Optionally add a small appendix table with:
     - Rows: Teacher, DMPD.
     - Columns: Acc/Macro-F1/ECE on test1 and avg(test2–5).

---

## C. Nice-to-have (only if time permits)

8. **Lightweight error / calibration analysis**
   - [x] Add 1–2 concrete error-case descriptions (text+image snippets) illustrating:
     - A case where the teacher is confidently wrong and gate/event-choice helps the student be more conservative.
   - [x] Briefly mention (in Analysis or Appendix) any observed correlation or calibration numbers; per-class ECE/NLL added for Weibo/WeFEND.

9. **Optional: add a simple bar plot based on existing tables**
   - [x] If space allows, create a simple bar plot (WeFEND or MiRAGeNews) comparing:
     - Text-only, SAFE, EANN, Teacher, DMPD.
     - For a single metric (e.g., Macro-F1 or ECE).
   - [x] Keep it strictly derived from table values (no new statistics), and place it in the appendix if needed.
