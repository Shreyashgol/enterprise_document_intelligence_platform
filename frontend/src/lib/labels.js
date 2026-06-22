// Color theme per entity label — used for highlighting and chips.
export const LABEL_STYLES = {
  PERSON: { chip: "bg-rose-100 text-rose-700", mark: "bg-rose-200/70" },
  ORG: { chip: "bg-indigo-100 text-indigo-700", mark: "bg-indigo-200/70" },
  EMAIL: { chip: "bg-sky-100 text-sky-700", mark: "bg-sky-200/70" },
  PHONE: { chip: "bg-cyan-100 text-cyan-700", mark: "bg-cyan-200/70" },
  DATE: { chip: "bg-amber-100 text-amber-700", mark: "bg-amber-200/70" },
  MONEY: { chip: "bg-emerald-100 text-emerald-700", mark: "bg-emerald-200/70" },
  LOCATION: { chip: "bg-fuchsia-100 text-fuchsia-700", mark: "bg-fuchsia-200/70" },
  PRODUCT: { chip: "bg-violet-100 text-violet-700", mark: "bg-violet-200/70" },
};

export const RELATION_STYLE = "bg-slate-100 text-slate-700";

export function labelChip(label) {
  return LABEL_STYLES[label]?.chip || "bg-slate-100 text-slate-600";
}

export function labelMark(label) {
  return LABEL_STYLES[label]?.mark || "bg-slate-200/70";
}
