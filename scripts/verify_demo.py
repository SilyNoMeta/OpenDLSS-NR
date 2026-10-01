"""Check --capture data with CPU reprojection and temporal-composition calculations.

    python scripts/verify_demo.py tmp/fox-compat [--compare tmp/fox-sm86] [--json tmp/demo.json]

This validates the frame plumbing, not the network: head values are observations. Full-network
references must still be checked with parity / compare_backends. No captures belong in public Git.
"""
from pathlib import Path
import argparse
import json
import numpy as np


def raw(path, dtype='<f4'):
    with open(path, 'rb') as f:
        w, h, c = np.fromfile(f, '<u4', 3)
        a = np.fromfile(f, dtype)
    if a.size != int(w) * int(h) * int(c):
        raise ValueError(f'truncated capture: {path}')
    return a.reshape(int(h), int(w), int(c))


def proxy(scene, paper):
    x = np.maximum(scene, 0).astype(np.float64) / max(paper, .05)
    x = np.where(x > .75, 1 - .25 * np.exp(-5.770780 * np.maximum(x - .75, 0)), x)
    x = np.clip(x, 0, 1)
    x = np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1/2.4) - .055)
    return x.astype(np.float16).astype(np.float64)


def bilinear(image, x, y):
    h, w = image.shape[:2]
    x = np.clip(x - .5, 0, w - 1); y = np.clip(y - .5, 0, h - 1)
    ix = x.astype(int); iy = y.astype(int)
    fx = (x - ix)[..., None]; fy = (y - iy)[..., None]
    jx = np.minimum(ix + 1, w - 1); jy = np.minimum(iy + 1, h - 1)
    return ((image[iy, ix] * (1-fx) + image[iy, jx] * fx) * (1-fy) +
            (image[jy, ix] * (1-fx) + image[jy, jx] * fx) * fy)


def history_sample(previous, motion):
    h, w = previous.shape[:2]; y, x = np.mgrid[:h, :w]
    pos = np.stack([x+.5, y+.5], -1) + motion[..., :2] * [w, h]
    base = np.floor(pos - .5) + .5; f = np.clip(pos-base, 0, 1)
    w0 = -.5*f + f*f - .5*f**3
    w1 = 1 - 2.5*f*f + 1.5*f**3
    w3 = -.5*f*f + .5*f**3; w2 = 1-w0-w1-w3; mid = w1+w2
    center = base + w2/mid; lo = base-1; hi = base+2
    taps = [(lo[..., 0], center[..., 1], w0[..., 0]*mid[..., 1]),
            (center[..., 0], lo[..., 1], w0[..., 1]*mid[..., 0]),
            (center[..., 0], center[..., 1], mid[..., 0]*mid[..., 1]),
            (center[..., 0], hi[..., 1], w3[..., 1]*mid[..., 0]),
            (hi[..., 0], center[..., 1], w3[..., 0]*mid[..., 1])]
    total = sum(bilinear(previous, px, py)*weight[..., None] for px, py, weight in taps)
    return total / sum(t[2] for t in taps)[..., None]


def verify(prefix):
    rows = []
    labels = ['before-reset','disabled','nr-moving','nr-off','nr-settled','nr-static',
              'pre-resize','reenabled','reset','resize-reset','resized']
    files = [prefix.parent / (prefix.name+'-'+label+'-frame.json') for label in labels]
    files = [p for p in files if p.exists()]
    if not files: raise ValueError(f'no captures for {prefix}')
    for path in files:
        stem = str(path)[:-len('-frame.json')]; m = json.loads(path.read_text())
        data = {n: raw(stem+'-'+n+'.raw') for n in ['scene', 'motion', 'head', 'previous', 'history', 'features']}
        for n, a in data.items():
            if not np.isfinite(a).all(): raise ValueError(f'{stem}: nonfinite {n}')
        scene, motion, head, prev, hist, features = (data[n] for n in data)
        h, w = scene.shape[:2]; y, x = np.mgrid[:h, :w]
        velocity = raw(stem+'-velocity.raw', '<u4')
        ndc = np.stack([(x+.5)/w*2-1, 1-(y+.5)/h*2], -1)
        rays = np.concatenate([ndc, np.full((h,w,1), .5), np.ones((h,w,1))], -1)
        clip = rays @ np.array(m['background']).reshape(4,4)
        movement = velocity[..., 2:4].copy().view('<f4').astype(float)
        sky = ((velocity[..., 0] == 0) & (velocity[..., 1] == 0)) | (velocity[..., 1].copy().view('<f4') == 0)
        movement[sky] = (clip[..., :2]/clip[..., 3:] - ndc)[sky]
        on_screen = (np.abs(ndc+movement) < 1).all(-1)
        expected_motion = movement * [.5, -.5]
        expected_motion[~on_screen] = 0
        expected_motion = expected_motion.astype(np.float16).astype(float)
        motion_error = np.abs(expected_motion-motion[..., :2])
        ulp = np.maximum(np.abs(np.spacing(expected_motion.astype(np.float16)).astype(float)), 1e-6)
        if np.any(motion_error > ulp*2) or not np.array_equal(on_screen, motion[..., 2] > .5):
            raise ValueError(f'{stem}: reprojection mismatch {motion_error.max()}')
        p = proxy(scene, m['paperWhite']); sample = history_sample(prev, motion)
        valid = on_screen & bool(m['historyValid'])
        neural = np.clip(p+head[..., :3]/4, 0, 1)
        weight = m['blendScale']/(1+np.exp(-np.clip(head[..., 3], -80, 80)))
        expected = np.where(valid[..., None], neural*(1-weight[..., None])+sample*weight[..., None], neural)
        if not m['enabled']: expected = p
        # RZ publication to half can move a value by one half step; texture interpolation is
        # allowed 8-bit subtexel precision. The fixed bounds cover that plus shader/CPU exp rounding.
        error = np.abs(expected-hist)
        limit = .004 if m['historyValid'] and m['enabled'] else .0011
        if error.max() > limit: raise ValueError(f'{stem}: history error {error.max()} > {limit}')
        if m['enabled']:
            history_input = sample.astype(np.float16).astype(float)
            history_input = (history_input-.5).astype(np.float16).astype(float)
            history_input = (history_input*.125).astype(np.float16).astype(float)
            centered = ((p-.5).astype(np.float16).astype(float)*.125).astype(np.float16).astype(float)
            history_input[~valid] = centered[~valid]
            input_error = np.abs(history_input-features[..., 7:10]).max()
            if input_error > .0006: raise ValueError(f'{stem}: history input error {input_error}')
        else: input_error = None  # preprocess is deliberately skipped in bypass
        name = stem[len(str(prefix))+1:]
        expected_flags = {'reset': (1,0), 'reenabled': (1,0), 'resize-reset': (1,0), 'disabled': (0,0)}
        if name in expected_flags and (m['enabled'],m['historyValid']) != expected_flags[name]:
            raise ValueError(f'{name}: reset/toggle flag mismatch {m}')
        rows.append(dict(capture=name, frame=m['frame'], backend=m['backend'], size=f'{w}x{h}',
                         history_valid=m['historyValid'], enabled=m['enabled'],
                         history_max_error=float(error.max()), history_input_max_error=input_error,
                         motion_max_error=float(motion_error.max()),
                         surface_motion_pixels=int(np.count_nonzero(~sky & (np.abs(movement).max(-1)>1e-5))),
                         nr_effect_mean=float(np.abs(hist-p).mean())))
        print(f'{name}: PASS history max {error.max():.6g}, motion max {motion_error.max():.6g}', flush=True)
    required = {'nr-static','nr-moving','reset','disabled','reenabled','resize-reset','resized'}
    if not required.issubset({r['capture'] for r in rows}): raise ValueError('incomplete scripted run')
    moving = next(r for r in rows if r['capture']=='nr-moving')
    if not moving['surface_motion_pixels']: raise ValueError('no surface motion observed')
    cameras = Path(str(prefix)+'-cameras.txt').read_text().splitlines()
    matrices = {}
    for i in range(0, len(cameras), 9):
        frame = int(cameras[i].split()[1])
        p = np.array([[float(v) for v in line.split()] for line in cameras[i+1:i+5]])
        v = np.array([[float(v) for v in line.split()] for line in cameras[i+5:i+9]])
        v[:3,3] = 0
        matrices[frame] = p@v
    independent_background = matrices[99]@np.linalg.inv(matrices[100])
    recorded = json.loads(Path(str(prefix)+'-nr-moving-frame.json').read_text())
    camera_error = np.abs(independent_background-np.array(recorded['background']).reshape(4,4).T).max()
    # Text camera matrices are stored with six significant digits by the demo.
    if camera_error > 2e-5: raise ValueError(f'camera reprojection matrix error {camera_error}')
    moving['camera_matrix_max_error'] = float(camera_error)
    if not any(r['nr_effect_mean'] > .001 for r in rows if r['enabled']): raise ValueError('no NR effect observed')
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('prefix', type=Path)
    ap.add_argument('--compare', type=Path); ap.add_argument('--json', type=Path)
    args = ap.parse_args(); report = dict(captures=verify(args.prefix))
    if args.compare:
        report['comparison'] = []
        for row in report['captures']:
            label = row['capture']; a = str(args.prefix)+'-'+label; b = str(args.compare)+'-'+label
            # Renderer inputs must be byte-identical before attributing differences to the backend.
            for kind in ['scene','motion','velocity']:
                if Path(a+'-'+kind+'.raw').read_bytes() != Path(b+'-'+kind+'.raw').read_bytes():
                    raise ValueError(f'{label}: renderer {kind} differs between runs')
            d = raw(a+'-history.raw').astype(float)-raw(b+'-history.raw')
            report['comparison'].append(dict(capture=label, identical=bool(np.all(d==0)),
                                              history_mean_abs_255=float(np.abs(d).mean()*255),
                                              history_max_abs_255=float(np.abs(d).max()*255)))
    if args.json: args.json.write_text(json.dumps(report, indent=2))
    print('PASS: frame plumbing; network accuracy and visual quality require separate validation.')


if __name__ == '__main__': main()
