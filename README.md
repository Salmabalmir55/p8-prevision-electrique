# P8 — Prévision de la consommation électrique et déploiement multi-environnements

Projet de fin de module DevOps / MLOps (Pr. Soufiane HAMIDA, 2026-2027), sujet **P8 — Séries
temporelles et déploiement multi-environnements**.

Un énergéticien veut prévoir la consommation électrique journalière. L'objectif du projet n'est pas
d'obtenir le meilleur modèle, mais une chaîne **reproductible, versionnée, containerisée et
automatisée**, déployée dans trois environnements (développement, préproduction, production), qu'un
tiers peut relancer depuis un clone du dépôt.

## Équipe

| Membre | Responsabilités principales |
|---|---|
| Salma | Données, DVC, MLflow, image Docker du pipeline, suivi des erreurs, Grafana |
| Houda | Modèles, évaluation, API FastAPI, image Docker de service, Prometheus, CI |

## Outils

Git/GitHub · Docker et Docker Compose · DVC · MLflow · FastAPI · Prometheus · Grafana · GitHub Actions

## Structure prévue

```
src/          code du pipeline et de l'API
tests/        tests unitaires et de validation du modèle
configs/      configuration par environnement (dev, staging, prod)
data/         données (versionnées par DVC, jamais dans Git)
models/       modèles produits par le pipeline (DVC)
reports/      métriques et figures
monitoring/   Prometheus et Grafana
docs/         rapport, guide de déploiement, présentation
```

## Installation (poste de développement)

Python 3.11 ou 3.12 :

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows ; sous Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements.txt
```

Le README sera complété au fil du projet (pipeline, API, Docker, supervision).
