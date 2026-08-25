# Publish the prepared repository to GitHub

The canonical remote name is:

```text
rana-m-ahmed/ResearchWork-CropCop
```

## 1. Create the empty repository

Create a repository named `ResearchWork-CropCop` under `rana-m-ahmed`.

- Do not initialize it with a README, licence, or `.gitignore`.
- Keep it private during the final licence/privacy review.
- Change it to public only after the checks in `docs/GITHUB_SETTINGS.md` are complete.

## 2. Push the prepared history

From the prepared repository directory:

```bash
git remote add origin https://github.com/rana-m-ahmed/ResearchWork-CropCop.git
git push -u origin main
git push origin v0.1.0-bootstrap
```

A clone created from the supplied Git bundle contains the same commit history and tag.

## 3. Configure repository settings

Apply the settings in `docs/GITHUB_SETTINGS.md`, then confirm both workflows pass:

- `Validate public evidence`
- `Build arXiv paper`

## 4. Do not upload restricted artifacts

Do not add the source image corpus, checkpoints, raw logits, full forensic archives, or the PTE binary. Their hashes and public certificates are already represented in the repository.
