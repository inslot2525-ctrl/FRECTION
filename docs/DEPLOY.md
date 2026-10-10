# Deploying FRECTION

The app runs as one container: FastAPI serves both the API and the built
dashboard. Try it locally first if you have Docker:

```bash
docker build -t frection .
docker run -p 8000:8000 frection        # open http://localhost:8000
```

## Option 1 — Render (simplest, free)

1. Sign in at https://render.com with your GitHub account.
2. **New +** → **Blueprint** → choose the `FRECTION` repository.
3. Render reads `render.yaml` and builds the container. First build takes a few minutes.
4. Open the URL Render gives you.

Notes:
- The free plan has 512 MB of memory, so `render.yaml` builds **without
  PyTorch** (`WITH_GNN=0`). Everything works except the GNN risk score.
- Free services sleep after about 15 minutes without traffic. The first visit
  after that takes up to a minute to wake. Open the link shortly before a demo.

## Option 2 — Hugging Face Spaces (free, enough memory for the GNN)

1. Sign in at https://huggingface.co and create a **Space**: SDK **Docker**, blank template.
2. Add the Space as a second remote and push:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/frection
   git push space main
   ```
3. Spaces needs a short header at the top of the Space's `README.md`. Edit that
   file **on the Space** (not on GitHub, where the header would show as a table):

   ```yaml
   ---
   title: FRECTION
   sdk: docker
   app_port: 8000
   ---
   ```

This builds with PyTorch, so the GNN risk score is available.

## Optional: AI summaries

Set `ANTHROPIC_API_KEY` as an environment variable (Render) or a secret
(Spaces). Never commit the key. On a public demo anyone who opens the link can
spend your credit, so either leave it unset or set a low spending limit in the
Anthropic console.

## Things to know about a public demo

- Uploaded files are processed in memory and not stored, but they do pass
  through the host. Do not upload real customer data.
- Analysis results live in the process memory and are lost on restart.
