# Copilot Instructions for Clarity Template

These instructions help AI coding agents work effectively in this repository. They capture project-specific patterns, structure, and workflows so agents can be productive immediately.

## Big Picture
- **Purpose:** Static, minimalist website template for AI research showcases. Primary entry pages are `index.html` and `clarity.html`.
- **Structure:**
  - `assets/stylesheets/`: SCSS sources (`_master.scss`, `main.scss`) and compiled CSS (`main.css`, `main_free.css`).
  - `assets/scripts/`: Vanilla JS enhancements (`main.js` for slides/video controls, `navbar.js` for TOC sidebar).
  - `clarity/`: Component-specific CSS/JS/SCSS and images used by `clarity.html` and as patterns for `index.html`.
  - `assets/fontawesome-free-6.6.0-web/`: Local FontAwesome Free distribution; can be swapped to Pro as per `README.md`.
  - `assets/fonts/`: Optional licensed font files (Charter demo present). Font choice configured in `_master.scss`.
  - `assets/figures/`: Showcase images used in docs and site.
- **External deps:**
  - CDN delivered libraries: `highlight.js`, `MathJax v2`, `img-comparison-slider`.
  - jQuery (only used in `clarity/clarity.js`). No bundler; scripts load directly in HTML.

## Authoritative Files & Patterns
- **Entry HTML:**
  - `index.html`: Demonstrates page layout, dark/light theme toggles (via inline styles and classes), slide-show sections, code highlighting, MathJax, and optional TOC (`assets/scripts/navbar.js`).
  - `clarity.html`: Additional pattern examples (slideshow, comparison, selectors). Treat as a design guide; copy sections into your page as needed.
- **CSS/SCSS:**
  - Edit styles in `assets/stylesheets/_master.scss` (fonts, theme variables) and `assets/stylesheets/main.scss`. Compile to `main.css` or `main_free.css` based on font choice.
  - Project favours utility classes and semantic containers like `.container.blog.main`, `.blog-title`, `.no-cover`, `.white`, `.gray`, columns classes (e.g., `.columns-4`, `.columns-6`, `.columns-8`). Use these when adding sections.
- **JS behaviours:**
  - `assets/scripts/main.js`:
    - Slide tabs: clicking dots (`#slide-menu .dot`) toggles `.slide-content` visibility by index.
    - Video controls: functions `ToggleVideo(x)`, `SlowVideo(x)`, `FastVideo(x)`, `RestartVideo(x)` act on elements with classes `<x>-video` and update `<x>-msg`.
    - Carousel: auto-advancing slider expects `.container .slider` children with fixed width positioning. Buttons `#prev_btn` and `#next_btn` control.
  - `assets/scripts/navbar.js`:
    - Builds a responsive, hoverable TOC from `.container.blog.main h1/h2` headings. Adds a fixed nav with collapse/expand logic; visibility tied to scroll position after first `h1`.
  - `clarity/clarity.js`:
    - jQuery-driven dependent image selectors: adjusts `<select>` width and updates image `src` based on pair selections (`#image-selector{1..4}`) into elements `#parti` and `#parti2`.

## Conventions & Gotchas
- **No build tool assumed:** CSS/JS are linked directly. If you edit SCSS, ensure compiled CSS exists and is referenced by your HTML.
- **Fonts:** Default is free fonts via `assets/stylesheets/main_free.css`. To use licensed fonts, update `_master.scss` per `README.md` and link `assets/stylesheets/main.css` in your HTML.
- **Icons:** FontAwesome Free is linked by default (`assets/fontawesome-free-6.6.0-web/css/all.min.css`). To enable Pro, change the `<link>` to the Pro path (if present) and use appropriate icon classes.
- **Math & Code:** MathJax v2 config is defined inline in `index.html`. Code blocks use `highlight.js` with Foundation theme.
- **Section detection:** `navbar.js` relies on the document structure: headings must live inside `.container.blog.main` blocks. Keep this when adding sections for TOC to work.
- **Slider expectations:** The carousel in `main.js` expects a `.slider` container and two buttons with ids `prev_btn` and `next_btn`. Children are absolutely positioned with `left` offsets of `440px` steps.

## Typical Workflows
- **Local preview:** Serve the folder statically to test CDN and relative paths.
  - Example commands:
    - Python: `python3 -m http.server -b 127.0.0.1 8080`
    - Node (if installed): `npx serve .`
  - Open `http://127.0.0.1:8080/index.html`.
- **Toggle TOC feature:** Comment/uncomment `<script src="assets/scripts/navbar.js"></script>` in the `<head>` of your page as noted in `README.md`.
- **Switch themes:** Change background color on the title container and toggle `.white` or `.no-cover` classes on `.blog-title` per `README.md` examples.
- **Add a slideshow tab:** Duplicate a `.slide-content` block and add a corresponding `.dot` in `#slide-menu`. Visibility is index-based.
- **Use video controls:** Add videos with class `<prefix>-video` and a message element with id `<prefix>-msg`, then wire buttons to call the global functions in `main.js` with `<prefix>`.

## Editing Guidance for Agents
- Prefer modifying `index.html` or creating new pages modelled on `clarity.html`.
- When adding components, reuse existing classes; avoid introducing frameworks.
- If adding SCSS variables or new utilities, place them in `assets/stylesheets/_master.scss` and compile to the correct CSS output expected by the HTML.
- Keep external CDN scripts consistent with current versions used in `index.html` unless there is a specific need to update; verify compatibility if you do.
- If you introduce new navigation sections, ensure headings are inside `.container.blog.main` for `navbar.js` to pick them up.

## Key References
- `index.html`: end-to-end usage of template features.
- `clarity/clarity.js`, `assets/scripts/main.js`, `assets/scripts/navbar.js`: main interactive behaviours.
- `assets/stylesheets/_master.scss`, `assets/stylesheets/main_free.css`: typography, layout, and theme.
- `README.md`: visual guidelines, theme variants, font/icon switches, feature updates.

If anything here is unclear or missing (e.g., SCSS compile steps you prefer, or additional components you use), share details and I’ll refine these instructions. 