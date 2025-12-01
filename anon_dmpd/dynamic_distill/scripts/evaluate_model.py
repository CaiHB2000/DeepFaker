#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
try:
    import yaml
except ImportError as exc:
    raise ImportError("Install pyyaml to use evaluate_model.py") from exc
from transformers import AutoTokenizer, AutoImageProcessor
from dynamic_distill.src.data import build_datasets
from dynamic_distill.src.models import DynamicModalDistillationModel, TextEncoder, VisionEncoder
from dynamic_distill.src.models.encoders import EncoderOutput
from dynamic_distill.src.training.trainer import DynamicDistillationTrainer, TrainerConfig
from dynamic_distill.src.utils import expected_calibration_error
from dynamic_distill.scripts.train_mvp import build_collate_fn

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model(cfg: Dict[str, Any]) -> DynamicModalDistillationModel:
    m = cfg["model"]
    text_name = m.get("text_model")
    vision_name = m.get("vision_model")
    text_disabled = text_name in (None, "", "none", "null")
    vision_disabled = vision_name in (None, "", "none", "null")
    if text_disabled:
        class ZeroText(torch.nn.Module):
            def __init__(self, dim:int):
                super().__init__(); self.dim=dim; self.register_buffer("z", torch.zeros(1))
            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
                b = input_ids.shape[0] if input_ids is not None else (attention_mask.shape[0] if attention_mask is not None else 1)
                device = input_ids.device if input_ids is not None else (attention_mask.device if attention_mask is not None else self.z.device)
                rep = torch.zeros((b,self.dim), device=device)
                return EncoderOutput(representation=rep, sequence=None, extras={})
        text_encoder = ZeroText(m["encoder_dim"])
    else:
        text_encoder = TextEncoder(model_name=text_name, projection_dim=m["encoder_dim"], local_files_only=m.get("local_files_only",False))
    if vision_disabled:
        class ZeroVision(torch.nn.Module):
            def __init__(self, dim:int):
                super().__init__(); self.dim=dim; self.register_buffer("z", torch.zeros(1))
            def forward(self, pixel_values=None):
                b = pixel_values.shape[0] if pixel_values is not None else 1
                device = pixel_values.device if pixel_values is not None else self.z.device
                rep = torch.zeros((b,self.dim), device=device)
                return EncoderOutput(representation=rep, sequence=None, extras={})
        vision_encoder = ZeroVision(m["encoder_dim"])
    else:
        vision_encoder = VisionEncoder(model_name=vision_name, projection_dim=m["encoder_dim"], local_files_only=m.get("local_files_only",False))
    return DynamicModalDistillationModel(
        num_classes=m["num_classes"],
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        encoder_dim=m["encoder_dim"],
        classifier_hidden=m.get("classifier_hidden"),
        dropout=m.get("dropout",0.1),
    )

def evaluate(trainer: DynamicDistillationTrainer, loader, temperature: float):
    model = trainer.model.eval()
    device = next(model.parameters()).device
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss=0; correct=0; total=0
    num_classes = model.num_classes
    confusion = torch.zeros(num_classes,num_classes,device=device,dtype=torch.long)
    probs_list=[]; labels_list=[]; logits_list=[]
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            outputs = model(
                text_batch={k:v.to(device) for k,v in batch["text"].items()},
                vision_batch={k:(None if v is None else v.to(device)) for k,v in batch["vision"].items()},
            )
            logits = outputs["fusion_logits"] / temperature
            total_loss += ce(logits, labels).item()
            preds = logits.argmax(dim=-1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
            confusion.index_put_((labels, preds), torch.ones_like(labels, dtype=torch.long), accumulate=True)
            probs = torch.softmax(logits, dim=-1)
            probs_list.append(probs.cpu())
            labels_list.append(labels.cpu())
            logits_list.append(logits.cpu())
    avg_loss = total_loss/total
    acc = correct/total
    probs = torch.cat(probs_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    tp = torch.diag(confusion.float())
    prec = tp / confusion.float().sum(dim=0).clamp_min(1.0)
    rec  = tp / confusion.float().sum(dim=1).clamp_min(1.0)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    f1_macro = f1.mean().item()
    f1_pos = f1[1].item() if num_classes>1 else f1_macro
    ece = expected_calibration_error(probs, labels)
    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_pos": f1_pos,
        "ece": float(ece),
        "probs": probs,
        "labels": labels,
        "logits": torch.cat(logits_list, dim=0) if logits_list else None,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val","test","test1","test2","test3","test4","test5"], default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--test-csv", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_config(args.config)
    if args.test_csv:
        cfg['data']['test']['csv_file'] = args.test_csv
    tokenizer_cfg = cfg['tokenizer']; vision_cfg = cfg['vision_processor']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg['name'], local_files_only=tokenizer_cfg.get('local_files_only',False))
    image_processor=None
    if cfg['model'].get('vision_model') not in (None,"","none","null"):
        image_processor = AutoImageProcessor.from_pretrained(vision_cfg['name'], local_files_only=vision_cfg.get('local_files_only',False))
    collate_fn = build_collate_fn(tokenizer, image_processor, tokenizer_cfg.get('max_length',128), vision_cfg.get('image_size',224), augment_cfg=vision_cfg.get('augment'), text_dropout=tokenizer_cfg.get('word_dropout',0.0))
    data_module = build_datasets(cfg, collate_fn=collate_fn)
    loader = None
    if args.split == 'val':
        loader = data_module.val
    else:
        test_loader = data_module.test
        if isinstance(test_loader, dict):
            loader = test_loader.get(args.split, test_loader.get('test'))
        else:
            loader = test_loader
    if loader is None:
        raise ValueError(f"No loader for split {args.split}")

    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    trainer = DynamicDistillationTrainer(model=model, optimizer=torch.optim.Adam(model.parameters()), config=TrainerConfig())
    metrics = evaluate(trainer, loader, args.temperature)
    out = {
        'split': args.split,
        'temperature': args.temperature,
        'metrics': {k:v for k,v in metrics.items() if k not in ('probs','labels','logits')},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    # optionally save logits/probs for calibration\n    if args.save_logits:\n        logits = metrics.get('logits')\n        labels = metrics.get('labels')\n        if logits is not None and labels is not None:\n            import pandas as pd\n            df = pd.DataFrame(logits.numpy(), columns=[f'logit_{i}' for i in range(logits.shape[1])])\n            df['label'] = labels.numpy()\n            csv_path = args.output.with_suffix('.csv')\n            df.to_csv(csv_path, index=False)\n    print(json.dumps(out, indent=2))\n*** End Patch

if __name__ == "__main__":
    main()
