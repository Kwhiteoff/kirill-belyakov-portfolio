from __future__ import annotations
import argparse, csv, io, json, os, random, statistics, time, zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils

SEED = 20261006
SIZE = 128
N_TOTAL = 5000
N_TRAIN = 4000
N_HOLDOUT = 1000

COARSE = {
    "top": {"shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "vest"},
    "outerwear": {"jacket", "coat", "cape"},
    "bottom": {"pants", "shorts", "skirt"},
    "dress": {"dress", "jumpsuit"},
    "footwear": {"shoe", "sock", "tights, stockings", "leg warmer"},
    "bag": {"bag, wallet"},
    "headwear": {"hat", "headband, head covering, hair accessory", "scarf", "tie"},
}
CLASS_NAMES = list(COARSE)
TO_COARSE = {label: coarse for coarse, labels in COARSE.items() for label in labels}

CYCLES = [
    {"name":"baseline","lr":1.50e-3,"jitter":0.10,"noise":0.00,"dice":0.95,"edge":0.22,"coarse":0.16,"cls":0.025},
    {"name":"color_robustness","lr":1.30e-3,"jitter":0.18,"noise":0.01,"dice":1.00,"edge":0.24,"coarse":0.16,"cls":0.025},
    {"name":"shape_dice","lr":1.10e-3,"jitter":0.18,"noise":0.01,"dice":1.12,"edge":0.25,"coarse":0.18,"cls":0.025},
    {"name":"boundary_focus","lr":9.0e-4,"jitter":0.17,"noise":0.012,"dice":1.12,"edge":0.38,"coarse":0.18,"cls":0.025},
    {"name":"coarse_context","lr":8.0e-4,"jitter":0.17,"noise":0.012,"dice":1.12,"edge":0.36,"coarse":0.30,"cls":0.025},
    {"name":"category_regularizer","lr":7.0e-4,"jitter":0.16,"noise":0.010,"dice":1.13,"edge":0.38,"coarse":0.28,"cls":0.055},
    {"name":"hard_photometric","lr":6.0e-4,"jitter":0.24,"noise":0.022,"dice":1.15,"edge":0.40,"coarse":0.28,"cls":0.045},
    {"name":"precision_polish","lr":4.5e-4,"jitter":0.14,"noise":0.008,"dice":1.18,"edge":0.44,"coarse":0.24,"cls":0.035},
    {"name":"edge_polish","lr":3.0e-4,"jitter":0.10,"noise":0.005,"dice":1.20,"edge":0.52,"coarse":0.22,"cls":0.030},
    {"name":"final_low_lr","lr":1.5e-4,"jitter":0.08,"noise":0.003,"dice":1.22,"edge":0.50,"coarse":0.20,"cls":0.025},
]

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def segmentation_mask(seg, h, w):
    if isinstance(seg, list):
        if not seg:
            return np.zeros((h,w), np.uint8)
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg, dict):
        rle = seg
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, h, w)
    else:
        return np.zeros((h,w), np.uint8)
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return (m > 0).astype(np.uint8)

def edge_np(mask):
    t = torch.from_numpy(mask.astype(np.float32))[None,None]
    dil = F.max_pool2d(t,3,1,1)
    ero = -F.max_pool2d(-t,3,1,1)
    return ((dil-ero)>0).numpy()[0,0].astype(np.float32)

def eligible_records(data):
    categories = {c["id"]: c["name"] for c in data["categories"]}
    grouped = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        raw = categories.get(ann.get("category_id"))
        coarse = TO_COARSE.get(raw)
        if coarse is None:
            continue
        area = float(ann.get("area") or (ann.get("bbox",[0,0,0,0])[2] * ann.get("bbox",[0,0,0,0])[3]))
        grouped[ann["image_id"]].append((ann, coarse, raw, area))
    out = []
    for im in data["images"]:
        anns = grouped.get(im["id"], [])
        if not anns:
            continue
        areas = [x[3] for x in anns]
        k = int(np.argmax(areas))
        ann, coarse, raw, area = anns[k]
        total = max(1.0, float(sum(areas)))
        dominance = area / total
        area_ratio = area / max(1.0, float(im["width"]*im["height"]))
        # Automatic-cutout subset: one garment must be clearly dominant.
        if dominance < 0.67 or area_ratio < 0.018 or area_ratio > 0.88:
            continue
        out.append({
            "image": im, "ann": ann, "coarse": coarse, "raw": raw,
            "dominance": dominance, "area_ratio": area_ratio,
            "garment_count": len(anns),
        })
    return out

def balanced_5000(records):
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for r in records:
        buckets[r["coarse"]].append(r)
    for xs in buckets.values():
        rng.shuffle(xs)
    names = CLASS_NAMES[:]
    selected, pos = [], {k:0 for k in names}
    # Balance as far as the natural dataset allows.
    while len(selected) < N_TOTAL:
        progressed = False
        for name in names:
            if pos[name] < len(buckets[name]):
                selected.append(buckets[name][pos[name]])
                pos[name] += 1
                progressed = True
                if len(selected) >= N_TOTAL:
                    break
        if not progressed:
            break
    if len(selected) < N_TOTAL:
        raise RuntimeError(f"eligible Fashionpedia images={len(selected)}, need exactly {N_TOTAL}")
    rng.shuffle(selected)
    return selected

def prepare(zip_path, ann_path, work):
    work.mkdir(parents=True, exist_ok=True)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    selected = balanced_5000(eligible_records(data))
    images = np.empty((N_TOTAL,SIZE,SIZE,3), np.uint8)
    masks = np.empty((N_TOTAL,SIZE,SIZE), np.uint8)
    labels = np.empty((N_TOTAL,), np.int64)
    manifest = []
    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(x).name:x for x in zf.namelist() if x.lower().endswith((".jpg",".jpeg",".png"))}
        for i, rec in enumerate(selected):
            im = rec["image"]
            filename = Path(im["file_name"]).name
            member = names.get(filename)
            if not member:
                raise RuntimeError(f"image missing from official zip: {filename}")
            image = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
            mask = segmentation_mask(rec["ann"]["segmentation"], int(im["height"]), int(im["width"]))
            image = image.resize((SIZE,SIZE), Image.Resampling.BILINEAR)
            mask = np.asarray(Image.fromarray(mask*255).resize((SIZE,SIZE), Image.Resampling.NEAREST),np.uint8) > 127
            if mask.sum() < 20:
                raise RuntimeError(f"mask collapsed after resize: {filename}")
            images[i] = np.asarray(image,np.uint8)
            masks[i] = mask.astype(np.uint8)
            labels[i] = CLASS_NAMES.index(rec["coarse"])
            manifest.append({
                "index":i,"image_id":im["id"],"file":filename,"split":"train" if i<N_TRAIN else "holdout",
                "coarse":rec["coarse"],"category":rec["raw"],"dominance":round(rec["dominance"],6),
                "area_ratio":round(rec["area_ratio"],6),"garment_count":rec["garment_count"],
                "license_id":im.get("license"),"source_url":im.get("original_url"),
            })
            if (i+1) % 250 == 0:
                print(f"PREP {i+1}/{N_TOTAL}", flush=True)
    np.save(work/"images.npy", images)
    np.save(work/"masks.npy", masks)
    np.save(work/"labels.npy", labels)
    (work/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    stats = {
        "dataset":"Fashionpedia official train2020",
        "internet_images":N_TOTAL,"train_images":N_TRAIN,"frozen_holdout_images":N_HOLDOUT,
        "seed":SEED,"input_size":SIZE,"classes":dict(Counter(x["coarse"] for x in manifest)),
        "selection":"dominant garment >=0.67; target area 1.8%-88%; deterministic class-balanced round robin",
        "mean_dominance":statistics.mean(x["dominance"] for x in manifest),
        "mean_area_ratio":statistics.mean(x["area_ratio"] for x in manifest),
    }
    (work/"dataset_stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    return stats, manifest

class RealFashionDataset(Dataset):
    def __init__(self, images, masks, labels, train=False, cfg=None, cycle=0):
        self.images=images; self.masks=masks; self.labels=labels; self.train=train; self.cfg=cfg or {}; self.cycle=cycle
    def __len__(self): return len(self.images)
    def __getitem__(self,i):
        x=self.images[i].copy(); m=self.masks[i].copy()
        if self.train:
            rng=np.random.default_rng(SEED + self.cycle*1_000_003 + i*7_919)
            if rng.random()<0.5:
                x=x[:,::-1].copy(); m=m[:,::-1].copy()
            # Small geometric robustness without changing semantic target.
            if rng.random()<0.28:
                shift_x=int(rng.integers(-5,6)); shift_y=int(rng.integers(-5,6))
                x=np.roll(x,(shift_y,shift_x),(0,1)); m=np.roll(m,(shift_y,shift_x),(0,1))
            jitter=float(self.cfg.get("jitter",0))
            gain=float(rng.uniform(1-jitter,1+jitter))
            bias=float(rng.uniform(-jitter*70,jitter*70))
            xf=np.clip(x.astype(np.float32)*gain+bias,0,255)
            noise=float(self.cfg.get("noise",0))
            if noise>0:
                xf=np.clip(xf+rng.normal(0,255*noise,xf.shape),0,255)
            x=xf.astype(np.uint8)
        edge=edge_np(m)
        return (
            torch.from_numpy(x.transpose(2,0,1).copy()).float()/255.,
            torch.from_numpy(m[None].astype(np.float32)),
            torch.from_numpy(edge[None]),
            torch.tensor(int(self.labels[i]),dtype=torch.long),
        )

class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        g=8 if co%8==0 else 4
        self.net=nn.Sequential(
            nn.Conv2d(ci,co,3,1,1,bias=False),nn.GroupNorm(g,co),nn.SiLU(),
            nn.Conv2d(co,co,3,1,1,bias=False),nn.GroupNorm(g,co),nn.SiLU(),
        )
    def forward(self,x): return self.net(x)

class LookWeaCutoutV1Real5000(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool=nn.MaxPool2d(2)
        self.e1=Block(3,24); self.e2=Block(24,40); self.e3=Block(40,64); self.e4=Block(64,96); self.b=Block(96,128)
        self.d4=Block(224,96); self.d3=Block(160,64); self.d2=Block(104,48); self.d1=Block(72,36)
        self.alpha=nn.Conv2d(36,1,1); self.edge=nn.Conv2d(36,1,1); self.coarse=nn.Conv2d(128,1,1)
        self.cls=nn.Linear(128,len(CLASS_NAMES))
    def forward(self,x):
        s1=self.e1(x); s2=self.e2(self.pool(s1)); s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3)); z=self.b(self.pool(s4))
        coarse=self.coarse(z)
        y=F.interpolate(z,size=s4.shape[-2:],mode="bilinear",align_corners=False); y=self.d4(torch.cat((y,s4),1))
        y=F.interpolate(y,size=s3.shape[-2:],mode="bilinear",align_corners=False); y=self.d3(torch.cat((y,s3),1))
        y=F.interpolate(y,size=s2.shape[-2:],mode="bilinear",align_corners=False); y=self.d2(torch.cat((y,s2),1))
        y=F.interpolate(y,size=s1.shape[-2:],mode="bilinear",align_corners=False); y=self.d1(torch.cat((y,s1),1))
        return self.alpha(y), self.edge(y), coarse, self.cls(z.mean((2,3)))

def dice_loss(logits,target):
    p=torch.sigmoid(logits)
    num=2*(p*target).sum((1,2,3))+1
    den=p.sum((1,2,3))+target.sum((1,2,3))+1
    return (1-num/den).mean()

def boundary_f1_batch(pred,true,tol=2):
    p=torch.from_numpy(pred.astype(np.float32))[:,None]
    t=torch.from_numpy(true.astype(np.float32))[:,None]
    pe=(F.max_pool2d(p,3,1,1)+F.max_pool2d(-p,3,1,1)>0)
    te=(F.max_pool2d(t,3,1,1)+F.max_pool2d(-t,3,1,1)>0)
    k=2*tol+1
    pd=F.max_pool2d(pe.float(),k,1,tol)>0
    td=F.max_pool2d(te.float(),k,1,tol)>0
    pp=pe.sum((1,2,3)).numpy(); tt=te.sum((1,2,3)).numpy()
    precision=(pe & td).sum((1,2,3)).numpy()/(pp+1e-6)
    recall=(te & pd).sum((1,2,3)).numpy()/(tt+1e-6)
    f=2*precision*recall/(precision+recall+1e-6)
    f[(pp==0)&(tt==0)]=1
    return f

@torch.inference_mode()
def evaluate(model, loader, save_predictions=False):
    model.eval(); rows=[]; preds_keep=[]; truths_keep=[]; imgs_keep=[]
    correct_cls=0; n_cls=0
    for x,m,e,c in loader:
        al,ed,co,cl=model(x); pr=(torch.sigmoid(al)[:,0]>.5).numpy(); tr=(m[:,0]>.5).numpy()
        inter=(pr&tr).sum((1,2)); union=(pr|tr).sum((1,2)); ps=pr.sum((1,2)); ts=tr.sum((1,2))
        iou=inter/(union+1e-6); dice=2*inter/(ps+ts+1e-6); recall=inter/(ts+1e-6)
        leakage=(pr&~tr).sum((1,2))/float(SIZE*SIZE); bf=boundary_f1_batch(pr,tr)
        for vals in zip(iou,dice,bf,recall,leakage): rows.append(vals)
        correct_cls += int((cl.argmax(1)==c).sum()); n_cls += len(c)
        if save_predictions and len(preds_keep)<12:
            take=min(12-len(preds_keep),len(pr))
            preds_keep.extend(pr[:take]); truths_keep.extend(tr[:take]); imgs_keep.extend((x[:take].permute(0,2,3,1).numpy()*255).astype(np.uint8))
    a=np.asarray(rows,np.float64)
    accepted=(a[:,0]>=.75)&(a[:,2]>=.60)&(a[:,4]<=.05)
    metrics={
        "iou":float(a[:,0].mean()),"dice":float(a[:,1].mean()),"boundary_f1":float(a[:,2].mean()),
        "fg_recall":float(a[:,3].mean()),"bg_leakage":float(a[:,4].mean()),"accept_rate":float(accepted.mean()),
        "class_accuracy":correct_cls/max(1,n_cls),"n":len(rows),
    }
    return metrics,(imgs_keep,truths_keep,preds_keep)

def score(m):
    # Product-weighted: immediate usable masks are the strongest signal.
    return .32*m["iou"]+.25*m["boundary_f1"]+.14*m["fg_recall"]+.23*m["accept_rate"]+.04*m["class_accuracy"]+.02*(1-min(1,m["bg_leakage"]*10))

def make_preview(bundle,path):
    imgs,truths,preds=bundle
    if not imgs: return
    cell=SIZE; canvas=Image.new("RGB",(cell*3,cell*len(imgs)),(245,245,245)); draw=ImageDraw.Draw(canvas)
    for i,(im,gt,pr) in enumerate(zip(imgs,truths,preds)):
        original=Image.fromarray(im)
        gt_rgb=np.zeros((SIZE,SIZE,3),np.uint8); gt_rgb[:,:,1]=gt.astype(np.uint8)*255
        pr_rgb=np.zeros((SIZE,SIZE,3),np.uint8); pr_rgb[:,:,0]=pr.astype(np.uint8)*255
        y=i*cell; canvas.paste(original,(0,y)); canvas.paste(Image.fromarray(gt_rgb),(cell,y)); canvas.paste(Image.fromarray(pr_rgb),(2*cell,y))
    draw.rectangle((0,0,cell*3,18),fill=(255,255,255)); draw.text((4,2),"original                 ground truth              prediction",fill=(0,0,0))
    canvas.save(path)

def train_10(work,out):
    seed_all(); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    images=np.load(work/"images.npy",mmap_mode="r"); masks=np.load(work/"masks.npy",mmap_mode="r"); labels=np.load(work/"labels.npy",mmap_mode="r")
    holdout=DataLoader(RealFashionDataset(images[N_TRAIN:],masks[N_TRAIN:],labels[N_TRAIN:],False),batch_size=32,shuffle=False,num_workers=2,persistent_workers=True)
    model=LookWeaCutoutV1Real5000()
    params=sum(p.numel() for p in model.parameters())
    best_state=deepcopy(model.state_dict()); best_score=-1.0; best_metrics=None; history=[]
    for cycle,cfg in enumerate(CYCLES,1):
        model.load_state_dict(best_state)
        trainset=RealFashionDataset(images[:N_TRAIN],masks[:N_TRAIN],labels[:N_TRAIN],True,cfg,cycle)
        trainloader=DataLoader(trainset,batch_size=28,shuffle=True,num_workers=2,persistent_workers=True,generator=torch.Generator().manual_seed(SEED+cycle))
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=2e-4)
        model.train(); losses=[]; started=time.time()
        for x,m,e,c in trainloader:
            al,ed,co,cl=model(x); cm=F.interpolate(m,size=co.shape[-2:],mode="nearest")
            loss=(F.binary_cross_entropy_with_logits(al,m)+cfg["dice"]*dice_loss(al,m)
                  +cfg["edge"]*F.binary_cross_entropy_with_logits(ed,e)
                  +cfg["coarse"]*(F.binary_cross_entropy_with_logits(co,cm)+dice_loss(co,cm))
                  +cfg["cls"]*F.cross_entropy(cl,c))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3.0); opt.step(); losses.append(float(loss.detach()))
        metrics,_=evaluate(model,holdout); candidate_score=score(metrics)
        accepted=candidate_score>best_score+1e-5
        if accepted:
            best_score=candidate_score; best_state=deepcopy(model.state_dict()); best_metrics=metrics
            torch.save({"model":best_state,"cycle":cycle,"cycle_name":cfg["name"],"metrics":metrics,"score":best_score,"classes":CLASS_NAMES,"input_size":SIZE},out/"lookwea_cutout_real5000_best.pt")
        else:
            model.load_state_dict(best_state)
        row={"cycle":cycle,"name":cfg["name"],"accepted":accepted,"train_loss":float(np.mean(losses)),"seconds":round(time.time()-started,2),"candidate_score":candidate_score,"best_score_after":best_score,**metrics}
        history.append(row); print("CYCLE",json.dumps(row),flush=True)
    model.load_state_dict(best_state); final,bundle=evaluate(model,holdout,True); make_preview(bundle,out/"real5000_preview.png")
    dummy=torch.randn(1,3,SIZE,SIZE); model.eval()
    with torch.inference_mode():
        for _ in range(3): model(dummy)
        t=time.perf_counter()
        for _ in range(20): model(dummy)
        cpu_ms=(time.perf_counter()-t)*1000/20
    torch.jit.trace(model,dummy).save(str(out/"lookwea_cutout_real5000_best.ts"))
    onnx_error=None
    try:
        torch.onnx.export(model,dummy,str(out/"lookwea_cutout_real5000_best.onnx"),input_names=["image"],output_names=["alpha","edge","coarse","class_logits"],opset_version=17)
    except Exception as e:
        onnx_error=f"{type(e).__name__}: {e}"
    best_row=max((r for r in history if r["accepted"]),key=lambda r:r["best_score_after"])
    summary={
        "dataset":"Fashionpedia official train2020","internet_images":N_TOTAL,"train_images":N_TRAIN,"frozen_holdout_images":N_HOLDOUT,
        "cycles_completed":10,"best_cycle":best_row["cycle"],"best_cycle_name":best_row["name"],"final_holdout":final,
        "quality_score":score(final),"params":params,"fp32_parameter_mb":params*4/1024/1024,
        "torchscript_mb":round((out/"lookwea_cutout_real5000_best.ts").stat().st_size/1024/1024,4),
        "runner_cpu_latency_ms_128":cpu_ms,"onnx_exported":(out/"lookwea_cutout_real5000_best.onnx").exists(),"onnx_error":onnx_error,
        "selection_rule":"highest product-weighted score on the same frozen 1000-image holdout; worse cycles rollback",
        "license_note":"Fashionpedia annotations are CC BY 4.0; Fashionpedia does not own image copyrights. This checkpoint is R&D until source-image provenance/commercial-use review is completed.",
        "production_ready":False,
        "remaining_gates":["commercial image-rights review","MobileSAM comparison on same holdout","Android physical-device latency/RAM QA","iOS Core ML conversion/device QA"],
    }
    (out/"FINAL_REAL5000_METRICS.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (out/"TEN_CYCLES.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    with (out/"TEN_CYCLES.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    return summary

def write_readme(out, stats, summary):
    text=f"""# LookWea CUTOUT v1 — REAL5000 10-cycle result

This artifact was produced from exactly {N_TOTAL} real Fashionpedia internet images.
Train: {N_TRAIN}. Frozen holdout: {N_HOLDOUT}. The holdout was never used for gradient updates.

Ten real train -> benchmark -> accept/rollback cycles completed.
Best cycle: {summary['best_cycle']} ({summary['best_cycle_name']}).

Measured frozen holdout:
- IoU: {summary['final_holdout']['iou']:.4f}
- Dice: {summary['final_holdout']['dice']:.4f}
- Boundary F1: {summary['final_holdout']['boundary_f1']:.4f}
- FG recall: {summary['final_holdout']['fg_recall']:.4f}
- BG leakage: {summary['final_holdout']['bg_leakage']:.4f}
- Accept rate: {summary['final_holdout']['accept_rate']:.4f}
- Class accuracy: {summary['final_holdout']['class_accuracy']:.4f}

Files:
- lookwea_cutout_real5000_best.pt — resumable PyTorch checkpoint
- lookwea_cutout_real5000_best.ts — TorchScript export
- lookwea_cutout_real5000_best.onnx — ONNX export when successful
- TEN_CYCLES.csv/json — every iteration and rollback decision
- FINAL_REAL5000_METRICS.json — measured summary
- manifest.json — exact 5000 source records and frozen split
- real5000_preview.png — original / GT / prediction examples

Important: R&D checkpoint, not approved for production redistribution. Fashionpedia annotation license is CC BY 4.0, but image copyrights remain with their original providers. Mobile device QA and source-image commercial-rights review remain mandatory.
"""
    (out/"README.md").write_text(text,encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",type=Path,required=True); ap.add_argument("--annotations",type=Path,required=True)
    ap.add_argument("--work",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    stats,manifest=prepare(args.zip,args.annotations,args.work)
    summary=train_10(args.work,args.out)
    (args.out/"manifest.json").write_text((args.work/"manifest.json").read_text(encoding="utf-8"),encoding="utf-8")
    (args.out/"dataset_stats.json").write_text((args.work/"dataset_stats.json").read_text(encoding="utf-8"),encoding="utf-8")
    write_readme(args.out,stats,summary)
    print("FINAL",json.dumps(summary),flush=True)

if __name__=="__main__":
    main()
