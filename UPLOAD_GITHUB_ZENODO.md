# GitHub and Zenodo upload guide

## Before uploading

1. Open `PACKAGE_AUDIT.json` and confirm that `status` is `passed`.
2. Confirm that the included `LICENSE` file identifies the MIT License.
3. Upload the **contents** of this folder as the repository root. Do not upload
   the outer release-package directory or any original project directory.

## GitHub

1. Create a new public repository with no auto-generated README, license, or
   `.gitignore` because those files are already staged here.
2. Upload this package's contents, or use Git locally, and commit them to `main`.
3. Confirm that no raw-data, cache, checkpoint, or private-manifest directory is
   visible on GitHub.
4. Copy the final repository URL into the manuscript and `CITATION.cff`.

## Zenodo archival release

1. Sign in to Zenodo and enable the GitHub integration for the repository before
   publishing the first GitHub release.
2. On GitHub, create release tag `v1.0.0`, use a descriptive release title, and
   attach the provided ZIP as an optional convenience artifact.
3. Allow Zenodo to archive the release, then complete the metadata using the two
   authors and the manuscript title in `CITATION.cff`.
4. Add the resulting Zenodo DOI to the manuscript, cover letter, repository
   README, and `CITATION.cff`.
5. Keep the version-specific DOI for the submitted code release; the concept DOI
   may also be retained for later versions.

Do not submit the manuscript until the repository URL and archived DOI replace
all remaining repository placeholders in the submission files.
