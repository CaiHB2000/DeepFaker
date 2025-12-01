# Responsible NLP Checklist（草稿）

- Data provenance: Describe Weibo/Twitter collection sources, licenses, time spans; cite original datasets. Note any filtering and event aggregation rules.
- Annotation quality: Report annotator process, inter-annotator agreement if available; describe ambiguous cases.
- Bias/fairness risks: Assess topic/culture skews; analyze per-event performance; list mitigations (calibration, reliability gates, error analysis with sensitive terms masked).
- Privacy & safety: No personal identifiers are included; images textualized when necessary; release strictly adheres to licenses; provide takedown policy for demo materials.
- Reproducibility artifacts: Release code, YAML configs, seeds, and scripts; include table-generation script linking JSON summaries to LaTeX; document hardware and time.
