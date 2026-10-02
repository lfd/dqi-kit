#!/usr/bin/env bash
# Publish the current release as one standalone commit on the public GitHub
# repository.
#
#   scripts/publish_release.sh VERSION [--dry-run]
#
# The public repository (remote `lfd`, local branch `public`) is a chain of
# snapshot commits, one per release, with no internal history. This script
# turns the tree of the current HEAD into the next snapshot:
#
#   1. checks: clean tree, HEAD on main, CHANGELOG.md has a `## [VERSION]`
#      section, `public` is up to date with `lfd/main`
#   2. builds the release tree from HEAD, dropping paths marked
#      `export-ignore` in .gitattributes (same rule as `git archive`)
#   3. creates the snapshot commit with that tree and the previous public
#      release as its only parent; the message carries the changelog section
#      and a `Source-Commit:` trailer pointing at the internal commit
#   4. tags the snapshot `vVERSION` (public) and HEAD `release/VERSION`
#      (internal; the two tags need distinct names in this one clone). The
#      public commit and tag carry the author identity of the previous public
#      commit (the repository's release account), the internal tag yours.
#   5. pushes `public` to `lfd/main` with its tag, pushes the internal tag to
#      `origin`, and creates a GitHub release from the changelog section if
#      the `gh` CLI is available
#
# With --dry-run the script stops after step 2 and shows what the snapshot
# would contain. Nothing is pushed.
set -euo pipefail

cd "$(dirname "$0")/.."

version="${1:-}"
dry_run="${2:-}"
if [[ -z "$version" || ! "$version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "usage: $0 VERSION [--dry-run]    (VERSION like 0.2 or 0.2.1)" >&2
    exit 1
fi
if [[ -n "$dry_run" && "$dry_run" != "--dry-run" ]]; then
    echo "unknown argument: $dry_run" >&2
    exit 1
fi

public_remote="lfd"
public_repo="lfd/dqi-kit"
public_branch="public"
internal_remote="origin"
public_tag="v${version}"
internal_tag="release/${version}"

# --- 1. checks -------------------------------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
    echo "working tree is not clean" >&2
    exit 1
fi
if [[ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]]; then
    echo "HEAD must be on main (is $(git rev-parse --abbrev-ref HEAD))" >&2
    exit 1
fi
if ! grep -q "^## \[${version}\]" CHANGELOG.md; then
    echo "CHANGELOG.md has no '## [${version}]' section" >&2
    exit 1
fi
for tag in "$public_tag" "$internal_tag"; do
    if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
        echo "tag ${tag} already exists" >&2
        exit 1
    fi
done
git fetch -q "$public_remote"
if [[ "$(git rev-parse "$public_branch")" != "$(git rev-parse "${public_remote}/main")" ]]; then
    echo "local branch ${public_branch} differs from ${public_remote}/main; reconcile first" >&2
    exit 1
fi

source_commit="$(git rev-parse HEAD)"

# --- 2. release tree -------------------------------------------------------
tmp_index="$(mktemp)"
trap 'rm -f "$tmp_index"' EXIT
GIT_INDEX_FILE="$tmp_index" git read-tree HEAD
excluded="$(git ls-files -z | git check-attr --stdin -z export-ignore \
    | awk 'BEGIN{RS="\0"} NR%3==1{path=$0} NR%3==0 && $0=="set"{print path}')"
if [[ -n "$excluded" ]]; then
    echo "excluding (export-ignore):"
    sed 's/^/  /' <<<"$excluded"
    GIT_INDEX_FILE="$tmp_index" xargs -d '\n' git rm -q --cached -- <<<"$excluded"
fi
tree="$(GIT_INDEX_FILE="$tmp_index" git write-tree)"

if [[ "$tree" == "$(git rev-parse "${public_branch}^{tree}")" ]]; then
    echo "release tree is identical to the current public tree; nothing to publish" >&2
    exit 1
fi

echo "==> changes against the current public release:"
git --no-pager diff --stat "${public_branch}" "$tree" | tail -1

# changelog section: from '## [VERSION]' up to the next '## [' or the link block
notes="$(awk -v v="$version" '
    $0 ~ "^## \\[" v "\\]" {on=1; next}
    on && /^## \[/ {exit}
    on && /^\[[0-9]+\.[0-9]+.*\]: / {exit}
    on {print}' CHANGELOG.md | sed -e '/./,$!d' -e :a -e '/^\n*$/{$d;N;ba' -e '}')"

message="$(printf 'DQI-Kit %s\n\n%s\n\nSource-Commit: %s\n' "$version" "$notes" "$source_commit")"

if [[ "$dry_run" == "--dry-run" ]]; then
    echo
    echo "==> dry run: the snapshot commit would read"
    echo "------------------------------------------------------------"
    echo "$message"
    echo "------------------------------------------------------------"
    echo "tree ${tree}, parent $(git rev-parse --short "$public_branch"); nothing created or pushed"
    exit 0
fi

# --- 3. snapshot commit ----------------------------------------------------
# Author and committer of the public history: the identity of the previous
# public commit, so every release is attributed to the same account.
public_name="$(git log -1 --format=%an "$public_branch")"
public_email="$(git log -1 --format=%ae "$public_branch")"
as_public=(env GIT_AUTHOR_NAME="$public_name" GIT_AUTHOR_EMAIL="$public_email"
               GIT_COMMITTER_NAME="$public_name" GIT_COMMITTER_EMAIL="$public_email")
echo "==> public identity: ${public_name} <${public_email}>"

commit="$(printf '%s\n' "$message" | "${as_public[@]}" git commit-tree "$tree" -p "$public_branch")"
git update-ref "refs/heads/${public_branch}" "$commit" "$(git rev-parse "$public_branch")"
echo "==> created public commit $(git rev-parse --short "$commit") on ${public_branch}"

# --- 4. tags ---------------------------------------------------------------
"${as_public[@]}" git tag -a "$public_tag" "$commit" -m "DQI-Kit ${version}"
git tag -a "$internal_tag" "$source_commit" -m "DQI-Kit ${version} (internal source commit)

Published as ${public_tag} on github.com/${public_repo} (public commit ${commit})."
echo "==> tagged ${public_tag} (public) and ${internal_tag} (internal)"

# --- 5. publish ------------------------------------------------------------
git push "$public_remote" "${public_branch}:main" "refs/tags/${public_tag}"
git push "$internal_remote" "refs/tags/${internal_tag}"
echo "==> pushed ${public_branch} and ${public_tag} to ${public_remote}, ${internal_tag} to ${internal_remote}"

if command -v gh >/dev/null 2>&1; then
    printf '%s\n' "$notes" | gh release create "$public_tag" --repo "$public_repo" \
        --title "DQI-Kit ${version}" --notes-file - \
        && echo "==> GitHub release ${public_tag} created"
else
    echo "gh CLI not found; create the GitHub release by hand from tag ${public_tag} with the changelog section as notes"
fi

echo
echo "done: https://github.com/${public_repo}/compare/$(git describe --tags --abbrev=0 "${commit}^" 2>/dev/null || echo "v0.1")...${public_tag}"
