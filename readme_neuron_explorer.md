# OmniMouse Dataset Explorer — Deployment Guide

## What this is

An interactive visualization of the OmniMouse calcium imaging dataset: 2,282 imaging planes across 78 animals and 328 scans, totaling 2.6 million segmented neurons. Users can browse a zoomable cloud of scans, group by animal/scan/depth, switch between correlation, average, and mask image views, and zoom from the full dataset down to single-neuron resolution.

## Contents of this directory

```
explorer/
├── index.html                  (30 KB)   Single-file visualization (HTML + JS + CSS)
├── README.md                              This file
└── web_data/
    ├── manifest.json           (885 KB)   Minified metadata for all 2,282 nodes
    └── thumbs/                 (289 MB)   20,538 WebP thumbnail images
        ├── {node_id}_avg_64.webp          Average image, 64px short edge
        ├── {node_id}_avg_256.webp         Average image, 256px short edge
        ├── {node_id}_avg_full.webp        Average image, full resolution
        ├── {node_id}_corr_64.webp         Correlation image, 64px
        ├── {node_id}_corr_256.webp        Correlation image, 256px
        ├── {node_id}_corr_full.webp       Correlation image, full resolution
        ├── {node_id}_mask_64.webp         Segmentation mask composite, 64px
        ├── {node_id}_mask_256.webp        Segmentation mask composite, 256px
        └── {node_id}_mask_full.webp       Segmentation mask composite, full resolution
```

Total size: ~290 MB. No build step, no dependencies beyond a CDN link to D3.js v7.

## How to deploy on GitHub Pages

### Option A: Embed in existing site via iframe (recommended)

1. **Set up Git LFS** (the thumbs directory has 20,538 binary files):

```bash
cd your-gh-pages-repo
git lfs install
git lfs track "explorer/web_data/thumbs/*.webp"
git add .gitattributes
```

2. **Copy this entire `explorer/` directory** into the repo root:

```bash
cp -r /path/to/explorer .
```

3. **Add an iframe** wherever you want the explorer to appear in your site's `index.html`:

```html
<section id="explorer-section">
  <div class="container blog main">
    <h2>Dataset Explorer</h2>
    <p>Browse 2,282 imaging planes across 78 animals. Zoom from the full
       cloud to single-neuron resolution.</p>
    <iframe src="explorer/index.html"
            style="width:100%; height:80vh; border:1px solid #333; border-radius:4px;"
            loading="lazy"
            title="OmniMouse Dataset Explorer"></iframe>
  </div>
</section>
```

4. **Commit and push:**

```bash
git add explorer/ .gitattributes
git commit -m "Add interactive dataset explorer"
git push origin gh-pages
```

The explorer will be live within a few minutes.

### Option B: Standalone page

The explorer also works as a standalone page at `/explorer/index.html`. No iframe needed — just deploy the directory and link to it.

## External dependencies

The only external dependency is **D3.js v7**, loaded from CDN:

```html
<script src="https://d3js.org/d3.v7.min.js"></script>
```

This is ~33 KB gzipped and is widely browser-cached. No other frameworks, build tools, or server-side components are required.

## How it works

- **On page load**: fetches `manifest.json` (~32 KB gzipped), initializes a D3 force simulation with 2,282 nodes
- **Cloud view**: nodes render as colored dots/rectangles. No images fetched yet.
- **Zoom in**: as nodes grow on screen, thumbnail images are lazy-loaded at progressive resolution:
  - 30–160px on screen → 64px thumbnails (~1.4 KB each)
  - 160–500px → 256px thumbnails (~12 KB each)
  - 500px+ → full resolution (~26 KB each)
- **Image type toggle**: switches between correlation (default), average, and mask views. Each type loads independently on demand.
- **Grouping modes**: Random, By Animal, By Scan, By Depth — nodes rearrange with spring dynamics
- **Selection**: click any node to center and inspect it. Dropdowns navigate by animal/session/scan/depth.

## Data transfer estimates

| Visitor behavior | Estimated transfer |
|---|---|
| Page load, no interaction | ~40 KB |
| Casual browse (zoom in, pan around) | ~3 MB |
| Moderate exploration (zoom into several clusters, inspect a few scans) | ~7 MB |
| Deep exploration (all image types, ~30 scans at full-res) | ~19 MB |
| Theoretical maximum (every image at every tier) | ~236 MB |

**Initial page load is ~40 KB.** The vast majority of data is loaded lazily as the user zooms in.

### GitHub Pages bandwidth

At the 100 GB/month soft limit, the site can serve approximately:
- 35,000 casual visitors/month
- 14,000 moderate explorers/month
- 5,000 deep explorers/month

This is well within budget for a typical academic project page.

### Git LFS

Git LFS free tier: 1 GB storage + 1 GB bandwidth/month.
- Storage: 290 MB → fits in free tier
- Bandwidth: 1 GB/month ≈ 350 casual visits or 140 moderate visits before needing an upgrade ($5/month for 50 GB)

**Alternative**: if LFS bandwidth becomes a concern, move `explorer/web_data/thumbs/` to an external CDN (e.g., Cloudflare R2, free egress) and update the `DATA_BASE` path in `explorer/index.html` line ~82:

```js
DATA_BASE: 'https://your-cdn.example.com/omnimouse/',
```

## Browser compatibility

Requires a modern browser with:
- Canvas 2D (all major browsers)
- WebP image support (Chrome, Firefox, Safari 14+, Edge — 97% global coverage)
- ES2020 JavaScript (nullish coalescing `??=`, etc.)
- Fetch API

Tested on Chrome, Firefox, and Safari.

## Preprocessing methods (for transparency)

The images were generated from raw calcium imaging data with the following processing:

**Normalization:**
- Average images: percentile clipping [0.1, 99.9], linear scale to [0, 255] uint8
- Correlation images: percentile clipping [1.0, 99.0], linear scale to [0, 255] uint8
- Mask composites: max weight per pixel across all CNMF segmentation masks, clipped at 99.9th percentile of nonzero values

**Stitching:**
261 of the 2,282 nodes are composites of 2–4 imaging fields at the same depth. Fields are placed in a common canvas using microscope motor coordinates (field_x_um, field_y_um) and physical FOV dimensions. Non-covered regions are filled with the minimum image value.

**Thumbnails:**
Three resolutions per image: 64px short edge (WebP q=50), 256px short edge (WebP q=50), full resolution (WebP q=85). Lanczos resampling. Aspect ratio preserved.

## File manifest

- `index.html` — self-contained visualization (30 KB)
- `web_data/manifest.json` — node metadata (885 KB, ~32 KB gzipped)
- `web_data/thumbs/*.webp` — 20,538 image files across 3 types × 3 resolutions × 2,282 nodes (289 MB total)

## Troubleshooting

**Blank page**: check browser console for errors. Most likely `manifest.json` failed to load — verify the file exists at the correct relative path.

**Images not loading**: the page must be served over HTTP (not opened as a local file). Use `python -m http.server 8080` for local testing.

**Slow initial load**: the manifest is 885 KB raw but ~32 KB gzipped. GitHub Pages serves gzip automatically. If serving from another host, ensure gzip/brotli compression is enabled.
