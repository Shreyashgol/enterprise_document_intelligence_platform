import { labelChip } from "../lib/labels";

export default function EntityList({ entities }) {
  if (!entities?.length) {
    return <p className="text-sm text-slate-400">No entities found.</p>;
  }
  return (
    <ul className="flex flex-wrap gap-2">
      {entities.map((e, i) => (
        <li key={i} className={`chip ${labelChip(e.label)}`}>
          <span className="font-semibold">{e.text}</span>
          <span className="opacity-70">{e.label}</span>
        </li>
      ))}
    </ul>
  );
}
