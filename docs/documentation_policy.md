# Documentation and task-tracking policy

Status: active. This document defines the rules for `docs/`, `todo.org`, and
`todo_archive.org`. `CLAUDE.md` only points here; this file is the source of
truth for the system itself.

## Goal

Give Claude exactly the context a task needs, and no more. `CLAUDE.md` is
read every session; `docs/*` files are read on demand. Splitting reference
material this way keeps the always-loaded context small while keeping
detailed material available when relevant. This only works if the split
stays accurate — an out-of-date index or a stale document is worse than
none, because it misdirects rather than informs.

## Prose style

Applies to all markdown documents, code comments, and docstrings in this
project.

- Terse and technically precise. State the fact; omit hedging, filler, and
  restatement.
- Human-readable. Prefer complete sentences over dense notation, except in
  tables, code, and formulas where notation is clearer.
- No colloquialisms, idioms, or marketing language. Avoid words with more
  than one plausible reading in context.
- One canonical term per concept. If a concept has an established name (in
  the code, in the literature, or already chosen in `docs/terminology.md`),
  use exactly that name — do not introduce a synonym. If a term is
  ambiguous or overloaded in this codebase, resolve it: pick one meaning,
  record the resolution in `docs/terminology.md`, and use only the
  resolved sense going forward.
- Exception: `README.md` at the repository root is upstream, public-facing
  project copy (citations, badges, promotional framing) and is not subject
  to this style; do not edit it to match.

## `docs/` organization

- `docs/OVERVIEW.md` is the index of every active document in `docs/`: one
  row per file, stating its scope and current status. It must be updated
  in the same change that adds, retitles, repurposes, or archives a
  document — an index update is not a follow-up task, it is part of the
  edit.
- Filenames are lowercase, underscore-separated, and name the specific
  topic, sub-project, or investigation the file covers (e.g.
  `transition1x_position_preprocessing.md`), not a generic label
  (`notes.md`, `misc.md` are not acceptable).
- **Avoid redundant documents.** Before creating a new file, check
  `docs/OVERVIEW.md` for an existing document covering the same topic.
  Extend that document instead of starting a new one unless the new
  material is a genuinely distinct topic, investigation, or audience.
  When in doubt, extend.
- `docs/terminology.md` is the single glossary for project-specific and
  overloaded terms (currently: "fragment"). New entries are sections
  appended to this one file, not new files — the glossary must stay a
  single lookup point.
- `docs/archive/` holds documents that are no longer active reference
  material: superseded proposals, investigations whose findings have been
  fully folded into either the code or a newer document, or fixes that
  have both landed and no longer need their original rationale kept
  alongside the active set. Archiving is a `git mv`, not a deletion —
  history is preserved. Remove the archived file's row from the active
  table in `docs/OVERVIEW.md` and add it to that file's archive table
  instead. Do not archive a document without flagging the candidate and
  its reasoning first; archiving is a judgment call about whether the
  material is still needed, not a mechanical cleanup step.

## Task and decision tracking (`todo.org`, `todo_archive.org`)

Both files are Org-mode. `todo.org` holds everything active; `todo_archive.org`
holds closed entries moved out of `todo.org` once they no longer need to
stay in the active list (see archiving, below). Conventions live here, not
duplicated in either file.

**TODO keyword sequence:**
```
#+TODO: TODO(t) IN-PROGRESS(i) BLOCKED(b) | DONE(d) CANCELLED(c) SUPERSEDED(s)
```

**Entry kinds**, distinguished by an Org tag on the heading:
- `:task:` — concrete work to do or verify.
- `:decision:` — a choice made (or to be made) and its rationale. Use this
  for anything a future reader would otherwise have to reverse-engineer
  from a diff or a commit message — version/library choices, rejected
  alternatives, policy adoptions.

**Entry template:**
```org
* STATE Short, specific title                                     :task:
  :PROPERTIES:
  :CREATED:  [YYYY-MM-DD Day]
  :END:
  Rationale: why this entry exists / why this choice was made.
  <for decisions:> Alternatives considered: ..., rejected because ...
  <on completion, add:>
  CLOSED: [YYYY-MM-DD Day]
  Outcome: what actually happened, if it differs from the plan.
```

`CREATED` and `CLOSED` are inactive timestamps (`[...]`, not `<...>`) —
they are a record, not an agenda item. Every entry needs a `Rationale`
line; an entry without one does not explain itself to a future reader and
should not be filed until it does.

**Archiving:** move an entry to `todo_archive.org` once it is `DONE`,
`CANCELLED`, or `SUPERSEDED` *and* no longer relevant to active work (a
finding that shaped a decision still being acted on stays in `todo.org`
until that work concludes). Move the heading verbatim, including its
`PROPERTIES` drawer and `CLOSED` line — do not summarize or drop
information when archiving.

## Maintenance discipline

This system only earns its keep if it is kept current, so:

- Before starting nontrivial work, check `docs/OVERVIEW.md` for
  documents relevant to the task instead of re-deriving context that is
  already written down.
- Update the relevant document(s) and `todo.org` as part of finishing a
  task, not as a separate follow-up. A task is not done until its record
  is.
- When something is learned, decided, or fixed that isn't obvious from
  the code or git history, write it down in the appropriate document (or
  create one, per the rules above) rather than letting it live only in
  conversation.
- Periodically check for documents that have drifted out of date or been
  obviated by newer work, and flag them as archive candidates in
  `docs/OVERVIEW.md` rather than leaving stale material mixed in with
  current material.
