#!/usr/bin/env bash
# Рабочая копия под задачу. Главный worktree репозитория — боевой каталог, из
# него systemd запускает бота; в нём ничего не правят. Каждая задача живёт в
# .worktrees/<тип>-<имя> на своей ветке.
#
#   scripts/worktree.sh new <тип>/<имя>     новая ветка от свежего origin/main
#                                           или уже существующая ветка
#   scripts/worktree.sh link                подключить .venv и модели речи
#                                           в текущий worktree (EnterWorktree, Codex)
#   scripts/worktree.sh remove <тип>/<имя>  удалить worktree; отказ, если есть
#                                           незакоммиченное или незапушенное
set -euo pipefail

TYPES='feat|fix|content|docs|chore|ci|refactor|test|style'
PROD=$(git worktree list --porcelain | awk 'NR == 1 {print $2}')

die() { echo "worktree: $*" >&2; exit 1; }

link_env() {
    local dir=$1
    [[ $dir != "$PROD" ]] || die "боевой каталог не рабочая копия"
    # .venv и модели весят около гигабайта: рабочие копии берут их у прода.
    [[ -e $dir/.venv ]] || ln -s "$PROD/.venv" "$dir/.venv"
    mkdir -p "$dir/data"
    [[ -e $dir/data/models ]] || ln -s "$PROD/data/models" "$dir/data/models"
    git -C "$dir" config core.hooksPath .githooks
}

worktree_of() {
    git worktree list --porcelain | awk -v ref="refs/heads/$1" '
        /^worktree / {path = substr($0, 10)}
        $0 == "branch " ref {print path; found = 1}
        END {exit !found}'
}

cmd_new() {
    local branch=${1:-}
    [[ $branch =~ ^($TYPES)/[a-z0-9][a-z0-9._-]*$ ]] \
        || die "ветка называется <тип>/<имя>, тип: ${TYPES//|/, }"
    local dir="$PROD/.worktrees/${branch//\//-}"
    [[ ! -e $dir ]] || die "$dir уже существует"
    git -C "$PROD" fetch --quiet --prune origin
    if git -C "$PROD" show-ref --verify --quiet "refs/heads/$branch"; then
        git -C "$PROD" worktree add --quiet "$dir" "$branch"
    elif git -C "$PROD" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
        git -C "$PROD" worktree add --quiet --track -b "$branch" "$dir" "origin/$branch"
    else
        git -C "$PROD" worktree add --quiet --no-track -b "$branch" "$dir" origin/main
    fi
    link_env "$dir"
    echo "$dir"
}

cmd_link() {
    link_env "$(git rev-parse --show-toplevel)"
}

cmd_remove() {
    local branch=${1:-} dir unpushed
    [[ -n $branch ]] || die "укажи ветку"
    dir=$(worktree_of "$branch") || die "у ветки $branch нет worktree"
    [[ $dir != "$PROD" ]] || die "боевой каталог не удаляется"
    [[ -z $(git -C "$dir" status --porcelain) ]] \
        || die "в $dir незакоммиченные изменения"
    git -C "$PROD" fetch --quiet --prune origin
    if git -C "$PROD" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
        unpushed=$(git -C "$PROD" rev-list --count "origin/$branch..$branch")
    else
        unpushed=$(git -C "$PROD" rev-list --count "origin/main..$branch")
    fi
    [[ $unpushed == 0 ]] || die "в $branch незапушенных коммитов: $unpushed"
    # --force снимает только отказ из-за игнорируемого: симлинков, __pycache__.
    git -C "$PROD" worktree remove --force "$dir"
    # Влитую ветку локально не держим; невлитая остаётся до мержа PR.
    if git -C "$PROD" merge-base --is-ancestor "$branch" origin/main; then
        git -C "$PROD" branch --quiet -D "$branch"
    fi
    echo "worktree: $dir удалён"
}

case ${1:-} in
    new) cmd_new "${2:-}" ;;
    link) cmd_link ;;
    remove) cmd_remove "${2:-}" ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
