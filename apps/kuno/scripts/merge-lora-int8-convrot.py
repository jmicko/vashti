#!/usr/bin/env python3
"""Merge a ComfyUI-format LoRA into an INT8 ConvRot (int8_tensorwise, per-row scale) safetensors model.

Stored weights are q = round(T(W) / s_row), where T is the per-group product of symmetric normalized
H4 stages (an involution; see stable-diffusion.cpp model_io/int8_dense.h). The merge works in the
rotated basis: M = q * s_row + T(dW), then each row is requantized with s = max|M_row| / 127 and
seeded stochastic rounding, like ComfyUI's recalculated-scale requantize.
dW = multiplier * alpha / rank * B @ A. Every other tensor is copied byte for byte.

usage: merge_lora_int8_convrot.py BASE LORA OUT [--multiplier 1.0] [--check N]
"""
import argparse, hashlib, json, multiprocessing, struct, sys
import numpy as np


def header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read(f, base, entry):
    f.seek(base + entry['data_offsets'][0])
    raw = f.read(entry['data_offsets'][1] - entry['data_offsets'][0])
    dt = entry['dtype']
    if dt == 'BF16':
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(entry['shape'])
    return np.frombuffer(raw, dtype={'F32': np.float32, 'F16': np.float16, 'I8': np.int8}[dt]).reshape(entry['shape'])


def rotate(m, group):
    """Apply T along the last axis in place (each H4 stage is symmetric and orthonormal)."""
    rows, cols = m.shape
    assert cols % group == 0
    v = m.reshape(rows, cols // group, group)
    stride = 1
    while stride < group:
        w = v.reshape(rows, cols // group, group // (4 * stride), 4, stride)
        a, b, c, d = (w[:, :, :, i, :].copy() for i in range(4))
        w[:, :, :, 0, :] = (a + b + c - d) * .5
        w[:, :, :, 1, :] = (a + b - c + d) * .5
        w[:, :, :, 2, :] = (a - b + c + d) * .5
        w[:, :, :, 3, :] = (-a + b + c + d) * .5
        stride *= 4
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('lora'); ap.add_argument('out')
    ap.add_argument('--multiplier', type=float, default=1.0)
    ap.add_argument('--check', type=int, default=4, help='modules to verify against a float reference')
    ap.add_argument('--jobs', type=int, default=6)
    args = ap.parse_args()

    ctx = (args.base, args.lora, args.multiplier)
    bh, bbase = header(args.base)
    lh, _ = header(args.lora)
    fb = open(args.base, 'rb')
    modules = sorted({k[len('diffusion_model.'):].rsplit('.', 2)[0] for k in lh if k.endswith('.lora_A.weight')})
    owner = {}
    for mod in modules:
        owner[mod + '.weight'] = mod
        if mod + '.weight_scale' in bh: owner[mod + '.weight_scale'] = mod
    # Merge in file order so the writer consumes results as they arrive.
    first = {mod: min(bh[k]['data_offsets'][0] for k in (mod + '.weight', mod + '.weight_scale') if k in bh) for mod in modules}
    ordered = sorted(modules, key=first.get)

    meta = dict(bh.get('__metadata__', {}))
    meta['kuno_merged_lora'] = json.dumps({'lora_modules': len(modules), 'multiplier': args.multiplier,
                                           'requantize': 'per-row absmax/127, seeded stochastic rounding'})
    bh['__metadata__'] = meta
    hb = json.dumps(bh, separators=(',', ':')).encode()
    hb += b' ' * ((8 - len(hb) % 8) % 8)
    order = sorted((e['data_offsets'][0], k) for k, e in bh.items() if k != '__metadata__')
    pending, qmin, done = {}, 0, 0
    with multiprocessing.get_context('fork').Pool(args.jobs) as pool, open(args.out, 'xb') as o:
        results = pool.imap(merge_module, [(ctx, mod, i < args.check or mod.endswith('token_refiner.blocks.0.mlp.fc1')) for i, mod in enumerate(ordered)])
        o.write(struct.pack('<Q', len(hb))); o.write(hb)
        pos = 0
        for start, k in order:
            assert start == pos, (k, start, pos)
            e = bh[k]; n = e['data_offsets'][1] - e['data_offsets'][0]
            if k in owner:
                if k not in pending:
                    mod, data, mod_qmin, report = next(results)
                    assert mod == owner[k], (mod, k)
                    pending.update(data); qmin = min(qmin, mod_qmin); done += 1
                    if report: print(report, flush=True)
                    print(f'[{done}/{len(modules)}] {mod}', flush=True)
                data = pending.pop(k)
                assert len(data) == n, k
                o.write(data)
            else:
                fb.seek(bbase + start); left = n
                while left:
                    chunk = fb.read(min(left, 1 << 26)); o.write(chunk); left -= len(chunk)
            pos += n
    assert not pending and done == len(modules)
    print('wrote', args.out, len(modules), 'modules merged; original q min', qmin)


def merge_module(job):
    (base, lora, multiplier), mod, check = job
    bh, bbase = header(base)
    lh, lbase = header(lora)
    seed = int.from_bytes(hashlib.sha256(mod.encode()).digest()[:8], 'little')
    if bh[mod + '.weight']['dtype'] == 'BF16':
        # Unquantized BF16 linear: W + dW in f32, then seeded stochastic rounding to BF16.
        with open(base, 'rb') as fb, open(lora, 'rb') as fl:
            w = read(fb, bbase, bh[mod + '.weight'])
            A = read(fl, lbase, lh['diffusion_model.' + mod + '.lora_A.weight'])
            B = read(fl, lbase, lh['diffusion_model.' + mod + '.lora_B.weight'])
            alpha = float(read(fl, lbase, lh['diffusion_model.' + mod + '.alpha']).reshape(()))
        rank = A.shape[0]
        assert B.shape == (w.shape[0], rank) and A.shape[1] == w.shape[1], (mod, w.shape, A.shape, B.shape)
        dw = (B @ A) * np.float32(multiplier * alpha / rank)
        m = np.ascontiguousarray(w + dw, dtype=np.float32)
        bits = m.view(np.uint32).astype(np.uint64)
        noise = np.random.default_rng(seed).integers(0, 1 << 16, size=bits.shape, dtype=np.uint64)
        rounded = ((bits + noise) >> 16).astype(np.uint16)
        finite = np.isfinite(m)
        out = np.where(finite, rounded, (bits >> 16).astype(np.uint16))
        report = None
        if check:
            back = (out.astype(np.uint32) << 16).view(np.float32)
            report = (f'check {mod} (bf16): changed={float((out != (w.view(np.uint32) >> 16)).mean()):.4f} '
                      f'noise_rms/dW_rms={float(np.sqrt(((back - m) ** 2).mean() / (dw ** 2).mean())):.2f}')
        return mod, {mod + '.weight': out.tobytes()}, 0, report
    with open(base, 'rb') as fb, open(lora, 'rb') as fl:
        cfg = bh[mod + '.comfy_quant']
        fb.seek(bbase + cfg['data_offsets'][0])
        conf = json.loads(fb.read(cfg['data_offsets'][1] - cfg['data_offsets'][0]))
        assert conf.get('format') == 'int8_tensorwise' and conf.get('convrot'), (mod, conf)
        group = int(conf['convrot_groupsize'])
        q = read(fb, bbase, bh[mod + '.weight'])
        s = read(fb, bbase, bh[mod + '.weight_scale']).astype(np.float32).reshape(-1, 1)
        A = read(fl, lbase, lh['diffusion_model.' + mod + '.lora_A.weight'])
        B = read(fl, lbase, lh['diffusion_model.' + mod + '.lora_B.weight'])
        alpha = float(read(fl, lbase, lh['diffusion_model.' + mod + '.alpha']).reshape(()))
    rank = A.shape[0]
    assert B.shape == (q.shape[0], rank) and A.shape[1] == q.shape[1], (mod, q.shape, A.shape, B.shape)
    dw = (B @ A) * np.float32(multiplier * alpha / rank)
    m = q.astype(np.float32) * s
    m += rotate(dw, group)
    s_new = (np.abs(m).max(axis=1, keepdims=True) / 127.0).astype(np.float32)
    s_safe = np.where(s_new > 0, s_new, np.float32(1))
    # Seeded stochastic rounding (as ComfyUI's requantize does): the dequantized base sits exactly on
    # the grid and |dW| is far below one step, so round-to-nearest would discard most of the LoRA.
    u = np.random.default_rng(seed).random(m.shape, dtype=np.float32)
    q_new = np.clip(np.floor(m / s_safe + u), -127, 127).astype(np.int8)
    del u
    report = None
    if check:
        err = q_new.astype(np.float32) * s_new - m
        assert (np.abs(err) <= s_new * (1 + 1e-5) + 1e-12).all(), mod
        report = (f'check {mod}: changed={float((q_new != q).mean()):.4f} '
                  f'noise_rms/dW_rms={float(np.sqrt((err ** 2).mean() / (dw ** 2).mean())):.2f} '
                  f'old/new scale mean={float(s.mean()):.3e}/{float(s_new.mean()):.3e} '
                  f'|dW|/|W|={float(np.linalg.norm(dw) / np.linalg.norm(m - dw)):.5f}')
    return mod, {mod + '.weight': q_new.tobytes(), mod + '.weight_scale': s_new.tobytes()}, int(q.min()), report


if __name__ == '__main__':
    sys.exit(main())
