---
name: blender-scene
description: Create or revise Blender scenes from natural-language descriptions, render review images, and preserve the prompt, bpy source, .blend scene, and screenshots locally. Use for Blender modeling, materials, camera, lighting, or scene-generation requests.
---

# Blender Scene

Turn the user's description into a real Blender scene, render it, inspect the
result, and keep a reproducible local artifact bundle.

## Output location

Store every generated job under:

```text
~/Desktop/DemoAI-TrainingData/blender-scenes/<descriptive-job-name>/
```

Never put generated `.blend` files or renders in the code repository. Each job
must retain at least:

- `prompt.txt`: the user's original text without rewriting it;
- `scene.py`: the complete Blender Python source used for this result;
- one `.blend` file;
- one rendered PNG screenshot.

A JSON manifest is useful but optional.

## Workflow

1. Confirm Blender is available. On this Mac, prefer
   `/Applications/Blender.app/Contents/MacOS/Blender`; otherwise use `blender`
   from `PATH`.
2. Translate the prompt into concrete scene requirements: geometry, dimensions,
   materials, spatial relationships, camera direction, lighting, background,
   and rendering style. Infer ordinary artistic details when the request leaves
   them open.
3. Write a self-contained `bpy` script for this job. It must accept
   `--prompt` and `--output-dir` after Blender's `--` separator, construct the
   scene, set an active camera, save the `.blend`, and render a PNG.
4. Run the script with `scripts/run_scene.py`. This wrapper archives the exact
   prompt and source and rejects incomplete output bundles.
5. Inspect the PNG at original resolution. Revise and rerender when the main
   object is clipped, poorly framed, unreadable, visibly floating, incorrectly
   lit, or inconsistent with the prompt.
6. Show the final screenshot and provide links to the `.blend`, prompt, code,
   and render. Keep the review loop open when the user requested approval.

Use procedural materials and modeled geometry when they can satisfy the prompt
without external assets. Do not download third-party assets unless the user
requests them or they are necessary and authorized.

## Blender compatibility

The validated local version is Blender 5.2 LTS. Its Eevee engine identifier is
`BLENDER_EEVEE`. Do not assume `BLENDER_EEVEE_NEXT` exists. Prefer AgX color
management and convert display sRGB values to linear values before assigning
node colors.

Use soft area lights for product-style scenes, include restrained fill only
when needed for readable shadows, and keep camera placement faithful to verbal
directions such as left/right and front/back.

## Reusable resources

- Run a generated script with `scripts/run_scene.py`.
- Use `scripts/generate_wooden_table.py` as the validated example for procedural
  wood, beveled furniture geometry, look-at camera placement, area lighting,
  saving, and rendering. Adapt it instead of copying irrelevant table details.

Do not commit generated job data. Commit skill source only after the user has
reviewed the representative output when a review step was requested.
