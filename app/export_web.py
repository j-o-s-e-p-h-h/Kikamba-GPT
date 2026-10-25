import json
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb


def export_for_web(model, ckpt, sp_model_path, val_txt_path, out_dir, n_tests=200):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = ckpt["model"]
    tensors = []
    offset = 0
    chunks = []
    for name, t in state.items():
        if name.endswith("attn.mask") or name == "head.weight":
            continue
        a = t.detach().float().cpu().numpy().astype(np.float16)
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        offset += a.size
        chunks.append(a.ravel())
    np.concatenate(chunks).tofile(out / "model.bin")

    proto = pb.ModelProto()
    proto.ParseFromString(open(sp_model_path, "rb").read())
    if proto.trainer_spec.model_type != pb.TrainerSpec.BPE:
        print("WARNING: this tokenizer is not BPE; the web tokenizer only supports BPE.")
    norm = proto.normalizer_spec
    tokenizer = {
        "pieces": [[p.piece, p.score, p.type] for p in proto.pieces],
        "normalizer": norm.name,
        "add_dummy_prefix": norm.add_dummy_prefix,
        "remove_extra_whitespaces": norm.remove_extra_whitespaces,
        "escape_whitespaces": norm.escape_whitespaces,
        "byte_fallback": proto.trainer_spec.byte_fallback,
        "unk_id": proto.trainer_spec.unk_id,
        "eos_id": proto.trainer_spec.eos_id,
    }
    json.dump({"config": ckpt["config"], "tensors": tensors, "tokenizer": tokenizer},
              open(out / "model.json", "w", encoding="utf-8"), ensure_ascii=False)

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    text = open(val_txt_path, encoding="utf-8").read()
    docs = [d for d in text.split("\n\n") if d.strip()][:n_tests]
    tok_tests = [{"text": d, "ids": sp.encode(d)} for d in docs]

    ids = [sp.eos_id()] + sp.encode(docs[0])[:40]
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids], device=next(model.parameters()).device))
    last = logits[0, -1].float().cpu().numpy()
    json.dump({"tokenizer": tok_tests, "logits_ids": ids,
               "logits_last": [round(float(v), 4) for v in last]},
              open(out / "selftest.json", "w", encoding="utf-8"), ensure_ascii=False)

    size_mb = (out / "model.bin").stat().st_size / 1e6
    print(f"wrote {out}: model.bin ({size_mb:.1f} MB), model.json, selftest.json")
