# Image du pipeline (cible « train ») : exécute `dvc repro` dans un environnement identique
# sur toute machine. Construction multi-étapes : les dépendances sont installées dans un
# environnement virtuel, puis seul ce dernier est copié dans l'image finale.

ARG PYTHON_VERSION=3.11

# ---------- Socle commun ----------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app

# ---------- Dépendances du pipeline ----------
FROM base AS train-deps
RUN python -m venv /opt/venv
COPY requirements/ requirements/
RUN pip install -r requirements/train.txt

# ---------- Socle d'exécution : utilisateur non privilégié ----------
FROM base AS runtime
# UID/GID fixes et explicites : identiques d'une construction à l'autre.
RUN groupadd --gid 999 app && useradd --uid 999 --gid 999 --home-dir /app --no-create-home app

# ---------- Image du pipeline ----------
FROM runtime AS train
COPY --from=train-deps /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app .dvc/config .dvc/config
COPY --chown=app:app dvc.yaml dvc.lock params.yaml .dvcignore ./
RUN mkdir -p data/raw data/processed models reports \
    && chown -R app:app /app
USER app
# HOME accessible en écriture pour les caches de DVC et matplotlib ; USER pour DVC.
ENV HOME=/tmp \
    USER=app \
    MPLCONFIGDIR=/tmp/matplotlib
CMD ["dvc", "repro"]
