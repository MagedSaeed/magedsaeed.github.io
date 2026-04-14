---
name: pptx-from-file
description: Use when the user asks to generate a .pptx presentation from a source file (pdf, markdown, docx, txt, ipynb, or similar). Produces a content-aware, visually rich presentation (colored/tinted backgrounds, accent-driven dividers — never plain black-on-white by default) with numbered slides, point animations, and handled figures. Build scripts live in /tmp; only the .pptx lands in .presentations/.
---

# pptx-from-file

## Overview

Convert any document into a polished .pptx whose design is **informed by the content itself** — not a generic template slapped on top. The default aesthetic is **visually rich and confident out of the box**: accent side-rails on every content slide, full-bleed accent-filled section dividers, a two-panel title slide with decorative marks, two-tone headers with accent block-underlines, richer bullet markers, and a colored closing. You exercise judgment on structure, visuals, and tone; `template.py` provides the consistent building blocks and does the heavy lifting on fanciness — no per-slide tricks needed.

Core principle: **you are the curator, not a transcriber.** A good presentation distills; it does not dump.

**Never ship plain black-on-white slides unless explicitly asked.** The fancy rich style is the default (`style="rich"` — already the default in `Presentation()`). A restrained minimal look is **opt-in only** — activate it with `Presentation(..., style="minimal")` and only when the user explicitly asks for "minimal", "editorial", "black and white", "plain", or "pure text". See "Default visual richness" below.

## Inputs

- **source** — required path (pdf, md, markdown, txt, docx, ipynb, tex, html)
- **output** — optional path. Default: `<repo-root>/.presentations/<source-stem>.pptx`
- **theme** — optional: `light` (default) or `dark`

If the user gave only a filename, resolve against the current working directory. If `.presentations/` does not exist at the repo root, create it.

## Build scripts: use /tmp, never pollute the project

**Do not write `.py` build scripts into the user's project tree** (not in `.presentations/`, not at the repo root, not anywhere the user will see them). They are one-off scaffolding and clutter the workspace.

- Write the build script to `/tmp/build_<source-stem>_<short-id>.py` (or any `/tmp` path).
- Run it from `/tmp` with `python3 /tmp/build_*.py`.
- The only artifact that should land in the project is the `.pptx` file at the chosen output path.
- If a prior run left a build script in the project, remove it.

## Workflow

1. **Read and structure the source.** Parse to sections/headings. For PDFs, prefer `pdftotext -layout` for text and `pdfimages`/`pdftoppm` for figures. For markdown/txt, parse natively. For ipynb, read JSON and keep markdown cells plus figures from output cells.
2. **Read the content for design signals** (see "Content-aware design" below) before picking the theme, accent, and typography.
3. **Draft an outline first.** Produce a slide-by-slide outline (title, 3–6 bullets each) and show it to the user. **Ask for approval or edits** before building. Include your design choices (accent, tone, serif vs sans) with a short rationale tied to the content.
4. **Ask when puzzled.** See "When to ask" below.
5. **Handle figures.** See "Figures and tables" below.
6. **Build the presentation** using `template.py` helpers. Don't hand-craft raw shapes when a helper exists.
7. **Apply default point animations.** See "Animations" below.
8. **Report** the output path and slide count. Offer to iterate.

## Design philosophy

The template produces slides that read as **professionally designed** — not AI-generated, not templated. The richness comes from a consistent system of tonal variation, typographic hierarchy, and repeated signature motifs. All of this is **baked into `template.py` by default** — you do not need to stack per-slide tricks.

**The five principles.** Every presentation should embody these:

1. **Multi-tone palette, not mono-accent.** At least five working tones: background, ink (primary text), muted (secondary text), primary accent, secondary accent. The default Atlas Navy theme uses dark navy `#0F1B2E` bg, cream `#F5EFE3` ink, muted blue-grey `#8B9BB0`, amber `#F5A524` primary, teal `#3EC4B1` secondary. Variation is what makes slides feel alive.

2. **Typographic hierarchy does the heavy lifting.** Size + weight + color create layers: eyebrow (small caps, letter-spaced, primary accent) → header (large bold, ink, first word underlined in accent) → body bullets (medium, ink, with accent bar markers) → captions/footer (small, muted). Avoid gratuitous decoration; let type carry the design.

3. **Signature motifs repeated across layouts.** The Atlas template uses: trio of brand marks (amber square + teal bar + amber bar) on title/closing; ghosted huge numeral on section dividers; first-word underline on every content header; solid amber vertical-bar bullet markers; teal-tick footer with title + author on every content slide; "NN / TT" slide counter in amber top-right. These recurring marks give every slide a family resemblance.

4. **Restraint with purpose.** Accent colors cover roughly 5–10% of each slide — eyebrows, bullets, rules, counters, tiny marks. Never body text. Whitespace is generous. One primary accent + one secondary accent per presentation.

5. **Sans-serif with confidence.** Lato (or Calibri fallback) throughout, heavy weights for titles, regular for body. Mixing serif + sans is reserved for special cases and requires explicit reason.

**What you automatically get out of the box:**

- **Title slide** — trio of brand marks top-left, amber eyebrow, 60pt bold title, muted subtitle, amber-bar author block with role, date bottom-right.
- **Section dividers** — giant 380pt ghosted numeral in a slightly-lighter navy, amber bar + "PART" eyebrow, big bold title, muted subtitle on the right.
- **Content slides** — amber eyebrow, bold header with first-word accent-underline, vertical-bar bullets in amber, teal-tick footer, "NN / TT" counter. Optional right-side tinted callout panel via `callout_title=` / `callout_body=`.
- **Quote** — amber eyebrow + rule, amber ghost quotation mark, huge bold cream body, amber-bar attribution.
- **Closing** — amber square brand mark, 96pt cream text, amber rule, muted subtitle, decorative ghost wedge bottom-right, amber URL.

**Variation is required, not optional.** The philosophy holds across every presentation; the **palette and mood must vary per content** so the user does not see the same deck twice. `template.py` ships with preset theme variants for exactly this reason:

| Preset | Use for | Palette signature |
|---|---|---|
| `Theme.atlas_navy()` (default) | Tech, ML, research, developer, product | deep navy + amber + teal + cream |
| `Theme.forest()` | Nature, biology, sustainability, ecology | dark forest green + gold + terracotta |
| `Theme.plum()` | Creative, design, pitch, bold marketing | dark plum + coral + electric blue |
| `Theme.slate()` | Corporate, finance, formal business | cool slate + cobalt + bronze |
| `Theme.kraft()` | Academic humanities, history, literature | kraft paper + burnt orange + forest |
| `Theme.sand()` | Light-theme design, lifestyle, editorial | warm sand + deep navy + coral |

**You must pick a variant based on the source's domain and tone** — do not default to Atlas Navy for every presentation. Announce the choice in the outline step: *"Picking `Theme.forest()` because the source is about biodiversity — warm gold accent on dark forest green fits the domain."*

Additional variation levers without leaving the philosophy:

- **Accent swap on any preset**: `Theme.atlas_navy()` produces navy + amber; `Theme.default(accent="indigo", accent2="teal")` keeps the same structure but shifts the primary mood.
- **Light vs. dark**: `sand` and `kraft` are light-bg variants; the rest are dark. Match to content energy (dark = focused/technical; light = open/reflective).
- **Header font**: default is Lato (sans); for editorial-literary content, swap to a serif via `Theme.kraft()` and set `header_font="Georgia"`.

Do not fight the template. Pick a variant that fits the content, write good copy, let the design system do its job.

**Minimal style is opt-in only.** When the user says "minimal", "editorial", "plain", "black and white", or "pure text", construct `Presentation(theme=..., style="minimal")`. This is a deliberate step-down — only use it when explicitly requested.

## Content-aware design

Before picking any visual, read a representative sample of the source and infer:

- **Domain** — tech/ML → indigo accent, sans-serif body; humanities/literature → warm accent, serif headers; finance/corporate → forest or slate accent, clean sans; nature/biology → forest accent; creative/design → warm accent with more whitespace.
- **Tone** — formal/academic → restrained palette, narrow accent use, serif headers; casual/pitch → slightly looser layouts, bolder accent; tutorial → heavier use of two-column with code/diagrams.
- **Density** — if source is dense and technical, allow more bullets per slide (up to 6) and use two-column layouts often; if narrative, fewer bullets per slide (3–4) and more section dividers.
- **Existing visual identity** — if the source already has a color scheme (brand PDF, existing figures), sample dominant hues and pick a template accent that harmonizes, not clashes.

Surface the inferred choices in the outline step: *"I'm suggesting the indigo accent and serif headers because the source reads as formal/technical. Okay, or do you want something else?"*

## Readability is non-negotiable

The template has a dark-on-light (or light-on-dark) contrast by design. When building slides:

- Body text ≥ 20pt; bullets ≥ 22pt; headers ≥ 32pt; title slide hero ≥ 48pt.
- Max line length ≈ 60 characters. Break or split slides instead of shrinking type.
- Never put text directly on a busy image — use a 60–80% opacity overlay or move text to a solid band.
- Accent color is for ~5% of each slide (rules, numbers, tiny marks). Never for body text.
- Respect `template.py`'s safe margins — do not draw in the outer 0.5" on any slide.
- One accent color per presentation.

## When to ask the user

Batch related questions into one message. Ask — don't assume — when any of these are unclear:

- **Audience and length.** "5-minute lightning talk or 30-minute seminar?" decides depth and slide count.
- **Section boundaries** for long or unstructured sources.
- **Figure placement** when a figure could go in multiple places.
- **Missing figures.** If a slide clearly needs a diagram and none exists, ask: (a) skip, (b) leave a labeled placeholder, (c) generate/sketch, or (d) fetch from the web (see below).
- **Web images.** **Always ask before pulling figures from the internet.** Name the source and purpose; wait for approval.
- **Tone, branding, accent color** — offer 2–3 options from the palette with rationale.
- **Animation level** — default is simple point animations; ask if the content calls for something richer or should be static (see below).
- **Slide numbers** — on by default. Ask only if the user wants them off or styled differently.
- **Output location** only if the default `.presentations/` would clobber an existing file.

## Figures and tables

- **Extract from PDF.** Use `pdfimages -all` or rasterize pages with `pdftoppm -r 200`. Show thumbnails (file paths) to the user and ask which to include and where.
- **Markdown image refs.** Resolve relative paths against the source file's directory. If broken, flag and ask.
- **Tables — small** (≤6 rows, ≤4 cols): render as native pptx tables via the helper.
- **Tables — large or complex:** tables that render poorly in native pptx **are screenshottable.** Use `libreoffice --headless --convert-to png` for .docx tables, render markdown tables via pandoc → image, or crop the PDF page. Fall back to asking the user if in doubt.
- **Missing but warranted.** If a slide clearly needs a visual (architecture, flow, comparison) but none exists, place a labeled placeholder and list those slides in the final report so the user can supply visuals. Ask before fetching from the web.
- **Captions.** Every figure gets a small italic caption ("Fig. 1 — …").
- **Placement defaults.** Supporting figure → two-column (text left, figure right, 60/40). Hero visual → full-bleed figure slide. Small inline chart → bottom of a content slide.

## Animations

- **Default:** point-by-point appearance for bullets (simple "appear" entrance, per paragraph). Titles and images appear with the slide, no entrance animation. Calm, progressive reveal.
- **Ask to change** when content suggests otherwise:
  - Dense reference slides or appendix-style content → suggest **no animations** (let the viewer scan).
  - Pitch, talk, or teaching material with progressive reveals → offer **more expressive animations** (fade, fly-in from left, scale-in for figures).
  - Charts or comparisons with a "reveal-the-answer" moment → offer per-element animations on specific slides only.
- Implementation: `template.py` injects the animation XML into each slide. Call `presentation.set_animation(mode="points"|"none"|"expressive")` once, or override per slide with `slide.animation = ...`.

## Slide numbers

Numbered by default, small type, bottom-right, muted color. Title slide and closing slide are excluded. To disable, the user must ask — do not remove silently.

## Template: "Atlas"

Full implementation in `template.py`. Design summary:

- **Palette (light):** background `#FAFAF7`, ink `#111111`, muted `#6B6B68`, rule `#E2DFD7`.
- **Palette (dark):** background `#121212`, ink `#F2EFE8`, muted `#B8B5AC`, rule `#2A2A2A`.
- **Accents:** `warm` `#C2410C`, `forest` `#1F4D3F`, `indigo` `#3B3B7B`, `slate` `#3F3F46`, `plum` `#6B2C5F`, `teal` `#0F6E6E`.
- **Type:** headers Georgia (display serif) or Calibri bold (geometric sans, pick per content); body Calibri/Arial. OS-safe fallbacks.
- **Motif (rich, default):** full-height accent side-rail on content slides; accent block-underline under headers; triangle `▸` bullet markers in accent; accent dot beside each slide number.
- **Layouts (rich):**
  1. `title_slide` — two-panel layout: eyebrow + serif title on the left, tall accent panel on the right with decorative marks
  2. `section_divider` — full accent background, 180pt number, decorative rule, title in light ink
  3. `content` — accent rail + optional eyebrow + serif header + accent block-underline + triangle bullets
  4. `two_column` — text left, figure or tinted supporting panel right (58/42)
  5. `figure_full` — centered image with italic muted caption
  6. `quote` — narrow left accent panel holds huge quote mark; serif body + muted attribution
  7. `closing` — full accent background with giant serif text, decorative mark, trio of dots

Pass `style="minimal"` to `Presentation(...)` only when the user asks for plain/editorial.

Never exceed 6 bullets. Never more than ~12 words per bullet. Overflow → split into two slides.

## Implementation

Use `template.py` in this skill directory. It exports:

```python
from template import Presentation, Theme

# Dark theme, indigo accent — rich style is the default
pres = Presentation(theme=Theme.dark(accent="indigo"))
pres.set_animation(mode="points")   # default; or "none" / "expressive"

pres.title_slide(title="...", eyebrow="...", author="...", date="...")
pres.section_divider(number="01", title="...")
pres.content(title="...", bullets=["...", "..."], eyebrow="optional")
pres.two_column(title="...", bullets=[...], image_path="...", caption="...")
pres.figure_full(image_path="...", caption="...")
pres.quote(text="...", attribution="...")
pres.closing(text="Thank you", subtitle="optional")
pres.save(output_path)

# Opt into plain style only when user asks:
# pres = Presentation(theme=Theme.light(accent="indigo"), style="minimal")
```

Read `template.py` before generating — it has the full API, palette constants, layout math, and the animation XML helpers.

## Quick reference

| Step | Action |
|---|---|
| Parse source | `pdftotext -layout` for PDF; native for md/txt/ipynb |
| Infer design | Read content → pick accent/type; share rationale in outline |
| Outline | Draft slides + design choices, confirm with user before building |
| Figures | Extract, show thumbnails, ask which; ask before web fetch |
| Tables | Native for small; screenshot for complex |
| Build | Use `Presentation` helpers from `template.py` — never raw XML (except via helpers) |
| Animate | Default point-by-point; ask if content suggests otherwise |
| Numbers | On by default (except title + closing) |
| Output | `.presentations/<stem>.pptx` unless user specifies |

## Common mistakes

- **Dumping the source.** Slides are not paragraphs.
- **Generic template bolted onto content.** Read the content first; let it drive accent, type, tone.
- **Plain black-on-white by default.** Visual richness is the default; minimal is opt-in only.
- **Leaving build scripts in the project.** All `.py` scaffolding lives in `/tmp`, never in the repo.
- **Poor contrast or tiny type.** Readability beats cleverness, always.
- **Silently fetching web images.** Always ask.
- **Silently skipping figures.** Ask about placement if in doubt.
- **Mixing accent colors.** One accent per presentation.
- **Skipping the outline-approval step.** Users hate rebuilding a 30-slide presentation they never wanted.
