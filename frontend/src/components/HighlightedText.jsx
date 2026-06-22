import { labelMark } from "../lib/labels";

// Renders text with entity spans highlighted by label color.
// Entities have {text,label,start,end}; spans must be non-overlapping (the
// backend guarantees this).
export default function HighlightedText({ text, entities }) {
  if (!text) return null;
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  const parts = [];
  let cursor = 0;

  sorted.forEach((e, i) => {
    if (e.start < cursor) return; // skip any overlap defensively
    if (e.start > cursor) {
      parts.push(<span key={`t${i}`}>{text.slice(cursor, e.start)}</span>);
    }
    parts.push(
      <mark
        key={`e${i}`}
        title={e.label}
        className={`rounded px-0.5 ${labelMark(e.label)} text-slate-900`}
      >
        {text.slice(e.start, e.end)}
        <sub className="ml-0.5 align-super text-[9px] font-bold uppercase text-slate-500">
          {e.label}
        </sub>
      </mark>
    );
    cursor = e.end;
  });
  if (cursor < text.length) {
    parts.push(<span key="tail">{text.slice(cursor)}</span>);
  }

  return (
    <div className="whitespace-pre-wrap break-words leading-8 text-slate-800 dark:text-slate-200">
      {parts}
    </div>
  );
}
