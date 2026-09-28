#!/usr/bin/env bash
# Полная проверка: перед коммитом задачи, перед PR и перед выкладкой.
# Её же запускает CI — там нет .venv, и тесты локальной речи пропускаются.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
else
    PYTHON=${PYTHON:-python3}
    echo "check: .venv нет — тесты локальной речи будут пропущены" >&2
fi
# Сравнение с пустым деревом проверяет пробелы во всём проекте вместе с
# незакоммиченными правками, а не только в последнем диффе.
EMPTY_TREE=$(git hash-object -t tree /dev/null)

"$PYTHON" -m unittest discover -s tests
"$PYTHON" -m compileall -q english_bot tests
"$PYTHON" -m english_bot.content.validate
git diff --check "$EMPTY_TREE"
# Unit-файл ссылается на пути этой машины; на раннере CI их нет.
if [[ -z ${CI:-} ]]; then
    systemd-analyze --user verify deploy/english-tutor-bot.service
fi
echo "check: всё зелёное"
