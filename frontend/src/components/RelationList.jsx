export default function RelationList({ relations }) {
  if (!relations?.length) {
    return <p className="text-sm text-slate-400">No relationships found.</p>;
  }
  return (
    <ul className="space-y-2">
      {relations.map((r, i) => (
        <li
          key={i}
          className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-800/50"
        >
          <span className="font-semibold text-slate-800 dark:text-slate-100">{r.source}</span>
          <span className="rounded bg-brand-50 px-2 py-0.5 font-mono text-xs text-brand-700 dark:bg-brand-500/15 dark:text-brand-500">
            {r.relation}
          </span>
          <span className="font-semibold text-slate-800 dark:text-slate-100">{r.target}</span>
        </li>
      ))}
    </ul>
  );
}
