#!/usr/bin/env bash
# Выкладка на прод. Боевой каталог — главный worktree репозитория, всегда на
# main и без правок. Он перематывается на origin/main только после зелёного CI
# этого коммита и зелёной локальной проверки с .venv, затем сервис
# перезапускается. Не поднялся — каталог и сервис возвращаются назад.
#
#   scripts/release.sh status   расхождение прода с origin/main, CI и сервис;
#                               код выхода 0 — расхождения нет
#   scripts/release.sh deploy   выкатить origin/main
#   scripts/release.sh restart  перезапустить тот же код после смены .env;
#                               отката нет, код не менялся
#
# RELEASE_SERVICE и RELEASE_REF переопределяют сервис и цель — только для
# проверки самого скрипта на копии репозитория.
set -euo pipefail

SERVICE=${RELEASE_SERVICE:-english-tutor-bot.service}
TARGET_REF=${RELEASE_REF:-origin/main}
UNIT=deploy/english-tutor-bot.service
WORKFLOW=ci.yml
# Справка печатается из этого файла, а не из боевого каталога, куда ниже cd.
SELF=$(realpath "$0")
PROD=$(git worktree list --porcelain | awk 'NR == 1 {print $2}')
cd "$PROD"

die() { echo "release: $*" >&2; exit 1; }
short() { git rev-parse --short "$1"; }

ci_state() {
    gh run list --workflow "$WORKFLOW" --commit "$1" --limit 1 \
        --json status,conclusion \
        --jq 'if length == 0 then "none" else .[0] | "\(.status)/\(.conclusion)" end' \
        2>/dev/null || echo unknown
}

wait_ci() {
    local sha=$1 state started=$SECONDS
    while :; do
        state=$(ci_state "$sha")
        case $state in
            completed/success) echo "release: CI $(short "$sha") зелёный"; return ;;
            completed/*) die "CI $(short "$sha") не прошёл: $state" ;;
            none | unknown) (( SECONDS - started < 180 )) || die "для $(short "$sha") нет прогона CI: $state" ;;
        esac
        (( SECONDS - started < 1200 )) || die "CI $(short "$sha") не завершился за 20 минут"
        echo "release: жду CI $(short "$sha"): $state"
        sleep 20
    done
}

local_check() {
    # Проверка идёт в отдельной копии: файлы прода меняются только перед
    # самым перезапуском, а работающий процесс не подхватывает их на полпути.
    local sha=$1 dir="$PROD/.worktrees/release-$(short "$1")" ok=0
    git worktree add --quiet --detach "$dir" "$sha"
    ln -s "$PROD/.venv" "$dir/.venv"
    mkdir -p "$dir/data"
    ln -s "$PROD/data/models" "$dir/data/models"
    (cd "$dir" && scripts/check.sh) || ok=$?
    git worktree remove --force "$dir"
    (( ok == 0 )) || die "локальная проверка $(short "$sha") не прошла"
}

restart() {
    local since
    since=$(date '+%Y-%m-%d %H:%M:%S')
    systemctl --user restart "$SERVICE"
    # «Запущен как» пишет app.py после getMe: процесс не просто жив, а на связи.
    for _ in $(seq 45); do
        if journalctl --user -u "$SERVICE" --since "$since" -q --no-pager | grep "Запущен как" >/dev/null; then
            systemctl --user is-active --quiet "$SERVICE" && return 0
        fi
        sleep 1
    done
    return 1
}

switch_to() {
    local from=$1 to=$2
    # Дерево чистое, поэтому --keep ничего не теряет и откажет, если это не так.
    git reset --quiet --keep "$to"
    if ! git diff --quiet "$from" "$to" -- "$UNIT"; then
        systemctl --user daemon-reload
    fi
}

cmd_status() {
    local head target dirty drift=0
    git fetch --quiet --prune origin
    head=$(git rev-parse HEAD)
    target=$(git rev-parse "$TARGET_REF")
    dirty=$(git status --porcelain)
    echo "Боевой каталог: $PROD, ветка $(git branch --show-current || true)"
    echo "HEAD:         $(git log -1 --format='%h %s' HEAD)"
    echo "$TARGET_REF:  $(git log -1 --format='%h %s' "$target")"
    if [[ $head == "$target" ]]; then
        echo "Код прода совпадает с $TARGET_REF"
    else
        drift=1
        echo "РАСХОЖДЕНИЕ: прод отстаёт на $(git rev-list --count HEAD.."$target"), впереди на $(git rev-list --count "$target"..HEAD)"
    fi
    if [[ -n $dirty ]]; then
        drift=1
        echo "РАСХОЖДЕНИЕ: правки в боевом каталоге:"
        echo "$dirty"
    fi
    echo "CI $(short "$target"): $(ci_state "$target")"
    echo "Сервис: $(systemctl --user is-active "$SERVICE" || true), запущен $(systemctl --user show -p ActiveEnterTimestamp --value "$SERVICE")"
    return $drift
}

cmd_deploy() {
    local prev target
    [[ $(git branch --show-current) == main ]] || die "боевой каталог не на main"
    [[ -z $(git status --porcelain) ]] \
        || die "в боевом каталоге правки — выясни их происхождение: scripts/release.sh status"
    git fetch --quiet --prune origin
    prev=$(git rev-parse HEAD)
    target=$(git rev-parse "$TARGET_REF")
    if [[ $prev == "$target" ]]; then
        echo "release: прод уже на $(short "$target"), выкладывать нечего"
        return
    fi
    git merge-base --is-ancestor "$prev" "$target" \
        || die "$(short "$prev") не предок $(short "$target"): выкладка только перемоткой вперёд"
    wait_ci "$target"
    local_check "$target"
    switch_to "$prev" "$target"
    if restart; then
        echo "release: прод $(short "$prev") → $(short "$target"), сервис на связи"
        return
    fi
    journalctl --user -u "$SERVICE" -n 30 --no-pager -q >&2 || true
    switch_to "$target" "$prev"
    restart || die "откат на $(short "$prev") тоже не поднял сервис — нужен человек"
    die "$(short "$target") не поднялся, прод возвращён на $(short "$prev"); main впереди прода — исправь через PR"
}

# .env читается только при старте сервиса, поэтому смена модели или ключа
# требует перезапуска без нового коммита. Код не меняется, откатывать нечего:
# не поднялся — ошибка в конфигурации, и её видно в журнале.
cmd_restart() {
    [[ -z $(git status --porcelain) ]] \
        || die "в боевом каталоге правки — выясни их происхождение: scripts/release.sh status"
    if restart; then
        echo "release: сервис перезапущен на $(short "$(git rev-parse HEAD)"), на связи"
        return
    fi
    journalctl --user -u "$SERVICE" -n 30 --no-pager -q >&2 || true
    die "сервис не поднялся после перезапуска — проверь .env по журналу выше"
}

case ${1:-} in
    status) cmd_status ;;
    deploy) cmd_deploy ;;
    restart) cmd_restart ;;
    *) sed -n '2,14p' "$SELF" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
