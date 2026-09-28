# Разработка и выкладка

`scope`: ветки и рабочие копии, коммиты, общая проверка, CI, выкладка на прод, откат, контроль расхождения прода с Git
`source_of_truth`: `.github/workflows/ci.yml`, `scripts/check.sh`, `scripts/worktree.sh`, `scripts/release.sh`, `.githooks/pre-commit`, `.githooks/pre-push`, `.gitignore`, `deploy/english-tutor-bot.service`

## Назначение

Как изменение проходит путь от задачи до работающего бота. Устройство самого
сервиса, env, данные и внешние сервисы — в `docs/operations.md`; обязательный
порядок действий агента — в `AGENTS.md`, раздел «Git, worktree и выкладка».

## Факты

- Прод и репозиторий живут на одной машине. User-unit
  `english-tutor-bot.service` — симлинк на `deploy/english-tutor-bot.service` —
  запускает `.venv/bin/python -m english_bot` из `/home/milton/english`. Этот
  каталог — главный worktree репозитория: перезапуск сервиса исполняет то, что
  лежит в нём на диске, закоммичено это или нет.
- Поэтому боевой каталог всегда на `main`, без правок и совпадает с
  `origin/main`. Меняет его только `scripts/release.sh deploy`.
- Репозиторий `github.com/Miltonian66/english-lab` публичный, основная ветка —
  `main`.
- Задачи делаются в рабочих копиях `.worktrees/<тип>-<имя>`, по одной ветке на
  задачу. `.venv` и `data/models` в них — симлинки на боевые: около гигабайта
  не копируется, а тесты локальной речи идут так же, как на проде.
- CI — GitHub Actions `.github/workflows/ci.yml`: запускается на каждый push в
  любую ветку и вручную. Матрица Python 3.11 (минимум ядра) и 3.14 (Python прода),
  единственный шаг — `scripts/check.sh`. Новый push в ветку отменяет её
  незавершённый прогон; прогоны `main` не отменяются.
- В CI нет `.venv`, поэтому шесть тестов `tests/test_providers.py::LocalSpeechTests`
  там пропускаются. Полностью, вместе с ними, код проверяет `scripts/release.sh`
  на машине прода перед перезапуском.
- `main` на GitHub защищён: изменения только через PR, обязательны проверки
  `check (Python 3.11)` и `check (Python 3.14)`, правило действует и для
  администраторов; force-push и удаление `main` запрещены. Слитые ветки GitHub
  удаляет сам.
- Git-хуки лежат в `.githooks/` и подключены `core.hooksPath=.githooks`; настройку
  выставляет `scripts/worktree.sh`.

## Контракты

### Ветки и коммиты

- Ветка называется `<тип>/<имя>` и растёт от свежего `origin/main`. Типы:
  `feat`, `fix`, `content`, `docs`, `chore`, `ci`, `refactor`, `test`, `style`.
  Имя проверяет `scripts/worktree.sh new`.
- Коммит — законченный шаг задачи с сообщением `<тип>: <что сделано>`.
  Промежуточные коммиты обязательны: незакоммиченная работа не видна никому и
  теряется вместе с рабочей копией.
- PR сливается merge-коммитом (`gh pr merge --merge --delete-branch`) и только с
  зелёным CI. Промежуточные коммиты остаются в истории `main`, а
  `scripts/worktree.sh remove` видит ветку влитой. После squash- или
  rebase-мержа локальная ветка не считается влитой и остаётся, пока её не удалят
  вручную.
- `pre-commit` отклоняет коммит в боевом каталоге и в ветке `main`, `pre-push` —
  push в `main`. Хуки ловят ошибку ещё до сети; на GitHub то же правило держит
  защита `main`. Имена обязательных проверок совпадают с `name` задания в
  `ci.yml`: при переименовании задания защиту нужно обновить, иначе PR не
  смержится.

### Рабочие копии: `scripts/worktree.sh`

| Команда | Что делает |
|---|---|
| `new <тип>/<имя>` | `git fetch`, worktree в `.worktrees/<тип>-<имя>`: новая ветка от `origin/main` без upstream, существующая ветка — как есть; подключает `.venv`, модели и хуки |
| `link` | то же подключение для текущей копии, созданной не скриптом (`EnterWorktree` в Claude Code, Codex) |
| `remove <тип>/<имя>` | отказ при незакоммиченных правках или незапушенных коммитах; иначе удаляет копию, а влитую в `origin/main` ветку — и локально |

Незапушенные коммиты считаются относительно `origin/<ветка>`, а если её нет — от
`origin/main`.

### Общая проверка: `scripts/check.sh`

| Шаг | Команда |
|---|---|
| тесты | `unittest discover -s tests` на `.venv/bin/python`; без `.venv` — на `python3` с предупреждением |
| компиляция | `compileall -q english_bot tests` |
| контент | `python -m english_bot.content.validate` |
| пробелы | `git diff --check` против пустого дерева — весь проект вместе с незакоммиченным |
| unit-файл | `systemd-analyze --user verify deploy/english-tutor-bot.service`, кроме CI |

Успех — код 0 и последняя строка `check: всё зелёное`.

### Выкладка: `scripts/release.sh deploy`

Шаги идут строго по порядку, и любой отказ оставляет прод нетронутым:

1. Боевой каталог на `main`, а `git status --porcelain` пуст. Иначе отказ со
   ссылкой на `status`: чужие правки не коммитятся и не откатываются скриптом.
2. `origin/main` достижим от HEAD перемоткой вперёд.
3. У коммита есть прогон `ci.yml` с итогом `completed/success`. Незавершённый
   прогон скрипт ждёт до 20 минут. Если прогона нет, отказ через 3 минуты.
4. `scripts/check.sh` проходит в отдельной detached-копии
   `.worktrees/release-<sha>` с `.venv` прода: файлы работающего процесса до
   перезапуска не меняются.
5. `git reset --keep` на цель, `systemctl --user daemon-reload`, если менялся
   unit-файл.
6. `systemctl --user restart`. Успех — строка «Запущен как» в журнале сервиса
   (её пишет `app.py` после `getMe`) в течение 45 секунд.
7. Сервис не поднялся — каталог и unit возвращаются на прежний коммит, сервис
   перезапускается. Скрипт завершается ошибкой, `main` остаётся впереди прода.

`scripts/release.sh status` показывает HEAD прода, `origin/main`, правки в
боевом каталоге, итог CI и состояние сервиса. Код 0 — расхождения нет.

### Откат

- Штатный: `git revert` в новой ветке → PR → CI → `scripts/release.sh deploy`.
- Аварийный, если прод лежит и ждать CI нельзя:
  `git -C /home/milton/english reset --keep <sha>` и
  `systemctl --user restart english-tutor-bot.service`, затем штатный revert. До
  него `status` показывает расхождение, а `deploy` снова выкатил бы плохой
  коммит.

### Что не попадает в Git

| Путь | Что там |
|---|---|
| `.env` | секреты |
| `/data/` | база, голосовые, кэш синтеза, модели, выгрузки |
| `/.venv` | окружение локальной речи; в рабочих копиях — симлинк |
| `/.worktrees/`, `/.claude/worktrees/` | рабочие копии задач |
| `/local/` | расшифровки сессий агентов и прочие локальные файлы |
| `/.lavish/` | исходники страниц, опубликованных через lavish |

Страницы lavish и локальные файлы кладутся в боевой каталог (`.lavish/`,
`local/`), а не в рабочую копию: `worktree.sh remove` удаляет копию вместе с
игнорируемым содержимым.

## Карта кода

- CI: `.github/workflows/ci.yml`.
- Общая проверка: `scripts/check.sh`.
- Рабочие копии: `scripts/worktree.sh`.
- Выкладка и контроль расхождения: `scripts/release.sh`.
- Хуки: `.githooks/pre-commit`, `.githooks/pre-push`.
- Unit-файл: `deploy/english-tutor-bot.service`; его параметры описаны в
  `docs/operations.md`.

## Проверка

```bash
scripts/check.sh
scripts/release.sh status
bash -n scripts/*.sh .githooks/*
gh run list --workflow ci.yml --branch main --limit 1
gh api repos/Miltonian66/english-lab/branches/main/protection --jq '.required_status_checks.contexts'
```

Успех: `check: всё зелёное`; `status` завершается с кодом 0 и пишет «Код прода
совпадает с origin/main»; синтаксис скриптов без ошибок; последний прогон CI на
`main` — `completed success`; защита перечисляет обе проверки CI.

## Ограничения

- Staging нет: CI проверяет код, а не живого бота. Первая живая проверка — прод
  после выкладки.
- Выкладка запускается командой, а не событием CI: у раннера GitHub нет доступа
  к машине прода, self-hosted runner не настроен.
- Рабочие копии делят с продом `.venv` и модели. Установка пакета в `.venv` из
  рабочей копии меняет окружение прода сразу, минуя CI.
- Перезапуск снимает идущие фоновые задачи (см. ограничения в
  `docs/operations.md`).
- Откат кода не откатывает базу: `Storage.initialize` не отказывается от более
  новой схемы и записывает в `user_version` свою `SCHEMA_VERSION`. Откат через
  релиз с миграцией сначала проверяется на копии базы.
