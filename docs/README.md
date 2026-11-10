# Documentation build

This project's documentation is built by two tools, deployed as one site.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose: home, design roadmap, theory, notebooks | [mystmd](https://mystmd.org) | `docs/*.md`, `docs/design/`, `docs/theory/`, `docs/notebooks/` | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` | `/reference/` |

Build both and assemble them with:

```bash
make docs          # build + assemble into public/
make docs-serve    # assemble, then serve public/ locally
```

## Why the split

mystmd is much better at prose, with real cross-references, first-class notebook execution, exports to PDF/LaTeX and MyST's directive syntax. However, it has no autodoc equivalent, and there is no mature way to render Python docstrings into a MyST site today.

MkDocs + mkdocstrings already does that well, and it publishes a Sphinx-compatible `objects.inv`, which is what mystmd needs to cross-reference *into* it. So each tool does the half it is good at.

## Site-nav URLs must be absolute

The theme re-renders the site nav from the config it embeds for hydration, and prepends `BASE_URL` to any nav URL starting with `/`. So a nav entry that already carries the deployment prefix is doubled (`/plumax/plumax/reference/`) and 404s.

This bites only *after* hydration. The server-rendered HTML is correct, so `curl` and `verify_links` both see a healthy link. We keep `site.nav` URLs absolute (mystmd rejects `/reference/` and `reference/` anyway) and never rewrite them at build time, and `nav_base_url_problems` in `scripts/build_docs.py` fails the build if one becomes root-relative.

The trade-off is that a local preview's nav button points at the deployed site. In-page `xref:` links are unaffected, since the theme does not re-prefix those, so they are rewritten to `{BASE_URL}/reference/...` as normal.

## If the theme download is blocked

`myst build --html` fetches the site template as a zip from GitHub. Behind a
corporate proxy or a restrictive egress policy that request can fail with a
403 while ordinary git access still works. Clone the template and point at it
locally instead:

```bash
git clone --depth 1 https://github.com/myst-templates/book-theme.git /tmp/book-theme
```

Then set `site.template` in `docs/myst.yml` to `/tmp/book-theme` for that
build, and nothing else changes.

Note that the rendered pages still load KaTeX, Font Awesome and jupyter-matplotlib stylesheets from CDNs at view time. Without network access in the *browser*, maths renders doubled, because KaTeX ships an accessibility MathML copy that its stylesheet is responsible for hiding. That is a viewing artefact and has nothing to do with the build.

## Citations

`docs/references.bib` is registered as `project.bibliography` in `docs/myst.yml`.
Cite from any MyST page with ``{cite:p}`key` `` or ``{cite:t}`key` ``; an unknown
key fails `myst build --strict`.

## Cross-references from prose into the API

In any MyST page, link to an API object with the `xref:` protocol and the
object's full dotted path, as it appears on its API page:

```markdown
[`simulate_plume`](xref:api#plumax.gauss_plume.simulate_plume)
[`MetField`](xref:api#plumax.met.MetField)
```

A target that does not exist in the inventory fails `myst build --strict`, so
broken API links are caught on the pull request rather than in production.

## Known upstream issue: the `$` anchor abbreviation

A Sphinx inventory may record an object's anchor as the literal `$`, meaning
"the anchor is the object's own name". mystmd 1.11.0 lowercases the name when it expands that abbreviation, which breaks links into mkdocstrings' case-sensitive anchors:

| Inventory entry | mystmd emits | Correct? |
|---|---|---|
| `plumax.met.MetField` -> `met/#$` | `#plumax.met.metfield` | no |
| `plumax.gauss_plume.simulate_plume` -> `gauss_plume/#$` | `#plumax.gauss_plume.simulate_plume` | yes (already lower case) |

The failure is silent. The link resolves and the page loads, but the browser cannot find the anchor.

This does bite in plumax. Each sub-package is documented on its own page, so every inventory entry uses `$`, and any link to a class or other mixed-case name comes out lowercased. We have two safety nets for it in `scripts/build_docs.py`:

- `restore_anchor_case` repairs the lowercased anchors after the build, using
  the inventory as the source of truth. Delete it once mystmd fixes the
  expansion upstream.
- `verify_links` checks that every internal link in the assembled site resolves to a file that exists and, when it carries a fragment, to an anchor that is really in that file. It sees both generators' output at once, so it catches this class of bug regardless of cause, and we keep it either way.

Both operate on the static HTML. Anything the theme re-renders on hydration (the site nav above being the case that actually bit us) is invisible to them, which is why that one needs its own check.
