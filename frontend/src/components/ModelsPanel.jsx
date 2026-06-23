import Section from "./Section";

// The Phase 7 NER ladder: four models of increasing capability that share the
// same task, label set and (from 7B on) the same from-scratch CRF decoder, so
// they're trained by one pipeline and compared head-to-head. Each rung isolates
// ONE variable, so every quality delta is attributable to a single change.
const MODELS = [
  {
    id: "7A",
    name: "BiLSTM",
    arch: "Embedding → BiLSTM → Linear",
    decoder: "Softmax · argmax per token",
    features: "From-scratch word embeddings",
    adds: "Baseline — contextual tagging learned from our own corpus.",
    strength: 25,
    tone: "slate",
  },
  {
    id: "7B",
    name: "BiLSTM + CRF",
    arch: "Embedding → BiLSTM → CRF",
    decoder: "Viterbi · global best valid path",
    features: "Scratch embeddings + learned tag transitions",
    adds: "CRF models tag→tag transitions and bans invalid BIO (no O→I-PER).",
    strength: 50,
    tone: "brand",
  },
  {
    id: "7C",
    name: "BERT",
    arch: "Pretrained Encoder → Linear",
    decoder: "Softmax over subwords → word level",
    features: "Contextual subword embeddings (pretrained)",
    adds: "Swaps scratch embeddings for an encoder trained on billions of tokens.",
    strength: 80,
    tone: "brand",
  },
  {
    id: "7D",
    name: "BERT + CRF",
    arch: "Pretrained Encoder → CRF",
    decoder: "Viterbi over gathered word emissions",
    features: "Pretrained encoder + learned tag transitions",
    adds: "The capstone: strongest features and globally-consistent sequences.",
    strength: 88,
    tone: "emerald",
  },
];

const BAR = {
  slate: "bg-slate-400 dark:bg-slate-500",
  brand: "bg-brand-500",
  emerald: "bg-emerald-500",
};

export default function ModelsPanel() {
  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="card p-5">
        <Section title="The NER model ladder">
          <p className="text-sm leading-6 text-slate-600 dark:text-slate-300">
            Phase 7 builds <strong>four</strong> named-entity recognisers of
            increasing capability. They share the same task, the same BIO label
            set, and — from 7B onward — the <strong>same from-scratch CRF</strong>,
            so one training pipeline fits them all and they can be compared
            head-to-head. Each rung changes exactly <strong>one</strong> thing
            over the rung above it (add a CRF, or swap to a pretrained encoder),
            so every quality difference is attributable to a single, explainable
            change.
          </p>
        </Section>
      </div>

      {/* Comparison cards */}
      <div className="grid gap-4 md:grid-cols-2">
        {MODELS.map((m) => (
          <div key={m.id} className="card p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <span className="rounded-md bg-slate-100 px-2 py-0.5 font-mono text-xs font-semibold text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                    {m.id}
                  </span>
                  <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
                    {m.name}
                  </h3>
                </div>
                <p className="mt-2 font-mono text-[12px] text-brand-700 dark:text-brand-400">
                  {m.arch}
                </p>
              </div>
            </div>

            <dl className="mt-4 space-y-2 text-[13px]">
              <Row label="Decoder" value={m.decoder} />
              <Row label="Features" value={m.features} />
              <Row label="Adds" value={m.adds} />
            </dl>

            {/* Illustrative relative quality */}
            <div className="mt-4">
              <div className="mb-1 flex items-center justify-between">
                <span className="label">Relative quality</span>
                <span className="text-xs text-slate-400">illustrative</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                <div
                  className={`h-full rounded-full ${BAR[m.tone]}`}
                  style={{ width: `${m.strength}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* The headline insight */}
      <div className="card p-5">
        <Section title="The result worth explaining">
          <p className="text-sm leading-6 text-slate-600 dark:text-slate-300">
            Expect the CRF to help the <strong>BiLSTM substantially</strong> but
            help <strong>BERT noticeably less</strong>. A strong contextual
            encoder already implicitly captures much of the tag-transition
            structure the CRF was added to enforce — so once the features
            underneath are good enough, the CRF's marginal value shrinks.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <Delta
              from="BiLSTM"
              to="+ CRF"
              note="large gain — the softmax head had no notion of valid sequences"
              tone="emerald"
              size="+++"
            />
            <Delta
              from="BERT"
              to="+ CRF"
              note="small gain — the encoder already models most transitions"
              tone="slate"
              size="+"
            />
          </div>
          <p className="mt-4 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-500 dark:bg-slate-800/50 dark:text-slate-400">
            Bars above are illustrative of the architectural trend, not measured
            scores. For exact entity-level P/R/F1 on a fixed test split, run{" "}
            <code className="rounded bg-slate-200 px-1 font-mono dark:bg-slate-700">
              python -m scripts.compare_models
            </code>{" "}
            — it trains all four on the same split and seed and writes the
            four-way comparison table.
          </p>
        </Section>
      </div>

      {/* Build constraints */}
      <div className="card p-5">
        <Section title="Built from first principles">
          <ul className="space-y-1.5 text-[13px] text-slate-600 dark:text-slate-300">
            <li>
              · The <strong>CRF is implemented from scratch</strong> (forward
              algorithm + Viterbi) — no <code className="font-mono">pytorch-crf</code>.
              The exact same <code className="font-mono">CRF</code> class powers
              both 7B and 7D.
            </li>
            <li>
              · Pretrained encoders are used for their{" "}
              <strong>weights + tokenizer only</strong>; the classification head,
              CRF, and subword↔word alignment are our code.
            </li>
            <li>
              · Entity-level metrics are computed from scratch and shared by every
              model, so the comparison is apples-to-apples.
            </li>
          </ul>
        </Section>
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex gap-2">
      <dt className="w-20 shrink-0 text-xs font-semibold uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="text-slate-700 dark:text-slate-300">{value}</dd>
    </div>
  );
}

function Delta({ from, to, note, tone, size }) {
  const color =
    tone === "emerald"
      ? "text-emerald-600 dark:text-emerald-400"
      : "text-slate-500 dark:text-slate-400";
  return (
    <div className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
        <span className="font-mono">{from}</span>
        <span className="text-slate-400">→</span>
        <span className="font-mono">{to}</span>
        <span className={`ml-auto font-mono text-base font-bold ${color}`}>{size}</span>
      </div>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{note}</p>
    </div>
  );
}
