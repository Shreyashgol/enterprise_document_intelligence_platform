// Small labeled section wrapper used inside result cards.
export default function Section({ title, count, children }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <h3 className="label">{title}</h3>
        {count != null && (
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-500 dark:bg-slate-800 dark:text-slate-400">
            {count}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}
