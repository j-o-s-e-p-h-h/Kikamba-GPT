// Kikamba GPT in the browser: a SentencePiece BPE tokenizer and a GPT
// forward pass with a key/value cache, ported from models/gpt.py.

(function (root) {
  "use strict";

  const NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, BYTE = 6;

  class Tokenizer {
    constructor(spec) {
      this.spec = spec;
      this.pieces = spec.pieces.map((p) => p[0]);
      this.scores = spec.pieces.map((p) => p[1]);
      this.types = spec.pieces.map((p) => p[2]);
      this.lookup = new Map();
      this.byteIds = new Map();
      this.pieces.forEach((p, i) => {
        const t = this.types[i];
        if (t === NORMAL || t === USER_DEFINED) this.lookup.set(p, i);
        if (t === BYTE) this.byteIds.set(parseInt(p.slice(3, 5), 16), i);
      });
    }

    normalize(text) {
      const s = this.spec;
      let t = s.normalizer === "identity" ? text : text.normalize("NFKC");
      if (s.normalizer.startsWith("nmt")) t = t.replace(/\s/g, " ");
      if (s.remove_extra_whitespaces) t = t.replace(/ +/g, " ").replace(/^ | $/g, "");
      if (t.length === 0) return "";
      if (s.add_dummy_prefix) t = " " + t;
      if (s.escape_whitespaces) t = t.replace(/ /g, "▁");
      return t;
    }

    encode(text) {
      const syms = Array.from(this.normalize(text));
      while (syms.length > 1) {
        let best = -Infinity, at = -1;
        for (let i = 0; i < syms.length - 1; i++) {
          const id = this.lookup.get(syms[i] + syms[i + 1]);
          if (id !== undefined && this.scores[id] > best) {
            best = this.scores[id];
            at = i;
          }
        }
        if (at < 0) break;
        syms.splice(at, 2, syms[at] + syms[at + 1]);
      }
      const ids = [];
      for (const sym of syms) {
        const id = this.lookup.get(sym);
        if (id !== undefined) ids.push(id);
        else if (this.spec.byte_fallback) {
          for (const b of new TextEncoder().encode(sym)) ids.push(this.byteIds.get(b));
        } else ids.push(this.spec.unk_id);
      }
      return ids;
    }

    decode(ids) {
      let out = "";
      let bytes = [];
      const flush = () => {
        if (bytes.length) out += new TextDecoder().decode(new Uint8Array(bytes));
        bytes = [];
      };
      for (const id of ids) {
        const t = this.types[id];
        if (t === BYTE) { bytes.push(parseInt(this.pieces[id].slice(3, 5), 16)); continue; }
        flush();
        if (t === CONTROL) continue;
        out += t === UNKNOWN ? " ⁇ " : this.pieces[id];
      }
      flush();
      out = out.replace(/▁/g, " ");
      if (this.spec.add_dummy_prefix && out.startsWith(" ")) out = out.slice(1);
      return out;
    }
  }

  function halfToFloatTable() {
    const table = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      if (e === 0) table[h] = s * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) table[h] = f ? NaN : s * Infinity;
      else table[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return table;
  }

  function erf(x) {
    const s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function matvec(W, b, x, outDim, inDim, out) {
    for (let o = 0; o < outDim; o++) {
      let sum = b ? b[o] : 0;
      const row = o * inDim;
      for (let i = 0; i < inDim; i++) sum += W[row + i] * x[i];
      out[o] = sum;
    }
    return out;
  }

  function layerNorm(x, w, b, out) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (x[i] - mean) * (x[i] - mean);
    const inv = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
    return out;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class KikambaGPT {
    static async load(baseUrl = "") {
      const meta = await (await fetch(baseUrl + "model.json")).json();
      const buf = await (await fetch(baseUrl + "model.bin")).arrayBuffer();
      return new KikambaGPT(meta, buf);
    }

    constructor(meta, buffer) {
      this.cfg = meta.config;
      this.tokenizer = new Tokenizer(meta.tokenizer);
      const half = new Uint16Array(buffer);
      const table = halfToFloatTable();
      this.w = {};
      for (const t of meta.tensors) {
        const size = t.shape.reduce((a, b) => a * b, 1);
        const arr = new Float32Array(size);
        for (let i = 0; i < size; i++) arr[i] = table[half[t.offset + i]];
        this.w[t.name] = arr;
      }
    }

    newCache() {
      return { k: [], v: [], length: 0 };
    }

    step(token, cache) {
      const { n_embd: C, n_head: H, n_layer: L, vocab_size: V, block_size } = this.cfg;
      const hs = C / H, w = this.w, pos = cache.length;
      if (pos >= block_size) throw new Error("context is full");
      const x = new Float32Array(C);
      for (let i = 0; i < C; i++) x[i] = w["tok_emb.weight"][token * C + i] + w["pos_emb.weight"][pos * C + i];
      const h = new Float32Array(C), q = new Float32Array(C), att = new Float32Array(C), proj = new Float32Array(C);
      const ff1 = new Float32Array(4 * C), ff2 = new Float32Array(C);
      for (let l = 0; l < L; l++) {
        const p = `blocks.${l}.`;
        if (!cache.k[l]) { cache.k[l] = []; cache.v[l] = []; }
        layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"], h);
        matvec(w[p + "attn.query.weight"], null, h, C, C, q);
        cache.k[l].push(matvec(w[p + "attn.key.weight"], null, h, C, C, new Float32Array(C)));
        cache.v[l].push(matvec(w[p + "attn.value.weight"], null, h, C, C, new Float32Array(C)));
        const T = pos + 1, scale = 1 / Math.sqrt(hs), scores = new Float32Array(T);
        for (let hd = 0; hd < H; hd++) {
          const o = hd * hs;
          let max = -Infinity;
          for (let t = 0; t < T; t++) {
            const k = cache.k[l][t];
            let s = 0;
            for (let i = 0; i < hs; i++) s += q[o + i] * k[o + i];
            scores[t] = s * scale;
            if (scores[t] > max) max = scores[t];
          }
          let sum = 0;
          for (let t = 0; t < T; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
          for (let i = 0; i < hs; i++) att[o + i] = 0;
          for (let t = 0; t < T; t++) {
            const v = cache.v[l][t], a = scores[t] / sum;
            for (let i = 0; i < hs; i++) att[o + i] += a * v[o + i];
          }
        }
        matvec(w[p + "attn.proj.weight"], w[p + "attn.proj.bias"], att, C, C, proj);
        for (let i = 0; i < C; i++) x[i] += proj[i];
        layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"], h);
        matvec(w[p + "ff.net.0.weight"], w[p + "ff.net.0.bias"], h, 4 * C, C, ff1);
        for (let i = 0; i < 4 * C; i++) ff1[i] = 0.5 * ff1[i] * (1 + erf(ff1[i] / Math.SQRT2));
        matvec(w[p + "ff.net.2.weight"], w[p + "ff.net.2.bias"], ff1, C, 4 * C, ff2);
        for (let i = 0; i < C; i++) x[i] += ff2[i];
      }
      cache.length += 1;
      layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], h);
      return matvec(w["tok_emb.weight"], null, h, V, C, new Float32Array(V));
    }

    sampleFrom(logits, temperature, topK, rand) {
      const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, topK);
      const top = idx.map((i) => logits[i] / temperature);
      const max = top[0];
      const probs = top.map((v) => Math.exp(v - max));
      const total = probs.reduce((a, b) => a + b, 0);
      let r = rand() * total;
      for (let j = 0; j < idx.length; j++) { r -= probs[j]; if (r <= 0) return idx[j]; }
      return idx[idx.length - 1];
    }

    async generate(prompt, { maxTokens = 80, temperature = 0.8, topK = 50, seed = 0, onText = null } = {}) {
      const tok = this.tokenizer, eos = tok.spec.eos_id;
      const ids = [eos].concat(tok.encode(prompt));
      const room = this.cfg.block_size - ids.length;
      if (room <= 0) throw new Error("prompt is too long");
      const cache = this.newCache(), rand = mulberry32(seed);
      let logits;
      for (const id of ids) logits = this.step(id, cache);
      const out = ids.slice(1);
      for (let n = 0; n < Math.min(maxTokens, room); n++) {
        const next = this.sampleFrom(logits, temperature, topK, rand);
        if (next === eos) break;
        out.push(next);
        if (onText) { onText(tok.decode(out)); await new Promise((r) => setTimeout(r, 0)); }
        if (cache.length >= this.cfg.block_size) break;
        logits = this.step(next, cache);
      }
      return tok.decode(out);
    }

    selfTest(tests) {
      let passed = 0, firstFail = null;
      for (const t of tests.tokenizer) {
        const got = this.tokenizer.encode(t.text);
        if (got.length === t.ids.length && got.every((v, i) => v === t.ids[i])) passed++;
        else if (!firstFail) firstFail = { text: t.text.slice(0, 80), expected: t.ids.slice(0, 12), got: got.slice(0, 12) };
      }
      const cache = this.newCache();
      let logits;
      for (const id of tests.logits_ids) logits = this.step(id, cache);
      let maxDiff = 0, argJs = 0, argPy = 0;
      tests.logits_last.forEach((v, i) => {
        maxDiff = Math.max(maxDiff, Math.abs(v - logits[i]));
        if (logits[i] > logits[argJs]) argJs = i;
        if (v > tests.logits_last[argPy]) argPy = i;
      });
      return { tokenizerPassed: passed, tokenizerTotal: tests.tokenizer.length, firstFail, maxLogitDiff: maxDiff, sameTopToken: argJs === argPy };
    }
  }

  root.KikambaGPT = KikambaGPT;
  if (typeof module !== "undefined") module.exports = { KikambaGPT, Tokenizer };
})(typeof window !== "undefined" ? window : globalThis);
