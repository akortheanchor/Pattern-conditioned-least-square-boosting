"""
Deep-learning baselines in JAX (CPU): SRCNN (Dong et al., 2014) and a
VDSR-style residual CNN (Kim et al., 2016), in 1-D and 2-D variants.
Both receive a classical upsampled/zero-filled input and get the same exact
data-consistency projection as PC-LSB at inference (strictly helps them).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax


def _conv(x, w, nd):
    dn = ("NWC", "WIO", "NWC") if nd == 1 else ("NHWC", "HWIO", "NHWC")
    return jax.lax.conv_general_dilated(x, w, (1,) * nd, "SAME", dimension_numbers=dn)


def init(arch, nd, key):
    if arch == "srcnn":
        spec = [(9, 1, 64), (1, 64, 32), (5, 32, 1)]
    else:  # vdsr-lite
        spec = [(3, 1, 32)] + [(3, 32, 32)] * 5 + [(3, 32, 1)]
    params = []
    for k, ci, co in spec:
        key, sk = jax.random.split(key)
        shape = (k,) * nd + (ci, co)
        fan = k ** nd * ci
        params.append((jax.random.normal(sk, shape) * np.sqrt(2.0 / fan) * (0.1 if co == 1 else 1.0),
                       jnp.zeros(co)))
    return params


def apply(params, x, nd, arch):
    h = x
    for i, (w, b) in enumerate(params):
        h = _conv(h, w, nd) + b
        if i < len(params) - 1:
            h = jax.nn.relu(h)
    return x + h  # residual learning for both (stabilises SRCNN too)


class CNNBaseline:
    def __init__(self, arch, nd, steps=1500, batch=16, crop=None, lr=3e-4, seed=0):
        self.arch, self.nd, self.steps, self.batch = arch, nd, steps, batch
        self.crop = crop or (48 if nd == 2 else 256)
        self.lr, self.seed = lr, seed

    def fit(self, Xin, Xgt, log=None, Vin=None, Vgt=None, eval_every=250):
        """Adam + cosine schedule; if validation data are given, the checkpoint with the
        lowest validation MSE is kept (early stopping / model selection)."""
        Xin = np.asarray(Xin, np.float32); Xgt = np.asarray(Xgt, np.float32)
        key = jax.random.PRNGKey(self.seed)
        params = init(self.arch, self.nd, key)
        sched = optax.warmup_cosine_decay_schedule(0.0, self.lr, min(100, self.steps // 10), self.steps)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
        st = opt.init(params)
        nd, arch = self.nd, self.arch

        def loss(p, a, b):
            return jnp.mean((apply(p, a, nd, arch) - b) ** 2)

        @jax.jit
        def step(p, s, a, b):
            l, g = jax.value_and_grad(loss)(p, a, b)
            u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s, l

        rng = np.random.default_rng(self.seed)
        c = self.crop
        best = (np.inf, params, 0)
        if Vin is not None:
            Va = jnp.asarray(np.asarray(Vin, np.float32)[..., None]); Vb = jnp.asarray(np.asarray(Vgt, np.float32)[..., None])
            fwd = jax.jit(lambda p, x: apply(p, x, nd, arch))

            def vloss(p):
                return float(np.mean([float(jnp.mean((fwd(p, Va[i:i + 16]) - Vb[i:i + 16]) ** 2))
                                      for i in range(0, len(Va), 16)]))
        t0 = time.time()
        for it in range(self.steps):
            idx = rng.integers(0, len(Xin), self.batch)
            if nd == 1:
                o = rng.integers(0, Xin.shape[-1] - c + 1, self.batch)
                a = np.stack([Xin[i, j:j + c] for i, j in zip(idx, o)])[..., None]
                b = np.stack([Xgt[i, j:j + c] for i, j in zip(idx, o)])[..., None]
            else:
                oi = rng.integers(0, Xin.shape[-2] - c + 1, self.batch)
                oj = rng.integers(0, Xin.shape[-1] - c + 1, self.batch)
                a = np.stack([Xin[i, u:u + c, v:v + c] for i, u, v in zip(idx, oi, oj)])[..., None]
                b = np.stack([Xgt[i, u:u + c, v:v + c] for i, u, v in zip(idx, oi, oj)])[..., None]
            if Vin is not None and it % eval_every == 0:
                v = vloss(params)
                if v < best[0]:
                    best = (v, params, it)
            params, st, l = step(params, st, a, b)
            if log is not None and it % 500 == 0:
                log(f"  {self.arch} step {it} loss {float(l):.3e} ({time.time()-t0:.0f}s)")
        self._pred = jax.jit(lambda p, x: apply(p, x, nd, arch))
        if Vin is not None:
            v = vloss(params)
            if v < best[0]:
                best = (v, params, self.steps)
            params = best[1]; self.best_step = best[2]
            if log is not None:
                log(f"  {self.arch} early-stopping: best val MSE {best[0]:.3e} at step {best[2]}")
        self.params = params
        return self

    def predict(self, xin):
        x = jnp.asarray(np.asarray(xin, np.float32)[None, ..., None])
        return np.asarray(self._pred(self.params, x))[0, ..., 0].astype(np.float64)


def load_cnn(arch, nd, params):
    m = CNNBaseline(arch, nd)
    m.params = [(jnp.asarray(w), jnp.asarray(b)) for w, b in params]
    m._pred = jax.jit(lambda p, x: apply(p, x, nd, arch))
    return m
