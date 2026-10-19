# Translate — traduction vocale FR → EN en direct

> Client à lancer sur le PC du présentateur. Il diffuse vers le serveur [traduction_live](https://github.com/jmagro21/traduction_live).

Parlez en français, le texte anglais s'affiche en **sous-titres façon film** :
texte centré en bas de l'écran, blanc à contour noir, **fond transparent**
(les clics passent à travers vers votre présentation). La légende se construit
**mot à mot** pendant que vous parlez.

## Trois moteurs

| Moteur | Temps réel | Internet | Détail |
|---|---|---|---|
| `vosk` | ★★★ **mot à mot** | traduction seule | Reconnaissance en streaming locale (Vosk) + traduction réseau. |
| `google` | par phrase | requis | Reconnaissance Google précise, mais validée seulement à chaque pause. |
| `whisper` | live approx. | non (hors-ligne) | faster-whisper 100% local. Peut halluciner. |

Le mode **`auto`** (par défaut) utilise Whisper sur un GPU NVIDIA lorsqu'il en
trouve un. La reconnaissance et la traduction restent alors locales, sans la
latence ni les limitations des services de traduction gratuits.

## Lancer

### Linux (Ubuntu / Debian)

Depuis le dossier `translate/` :

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-tk libportaudio2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Dans `.env`, renseignez `LIVE_SERVER` (l'adresse du Live Server choisi, par
exemple `https://sous-titres.exemple.fr`) et `LIVE_API_KEY` avec **exactement la
clé définie pour Live Server dans le `.env` de Docker Compose**.
Compose transmet cette clé au serveur ; il ne la génère pas automatiquement.

```bash
python main.py --list-devices
python main.py --headless --engine vosk --device 0
```

Le micro appartient à la machine qui exécute Translate. Les traductions
intermédiaires et finales sont envoyées au serveur ; le son reste traité par
le moteur choisi. `--headless` n'utilise pas Tkinter et s'arrête avec Ctrl+C.
Il est activé automatiquement sous Linux si `DISPLAY` est absent. Avec un
bureau X11/XWayland, omettez cette option pour afficher aussi les sous-titres
locaux ; le fond est opaque sous Linux. La molette Linux est prise en charge.
Pour un fonctionnement local sans diffusion, laissez les deux variables vides.

Le serveur qui reçoit ces données est le projet **Live Server**
([traduction_live](https://github.com/jmagro21/traduction_live)). S'il tourne sur
la même machine, vous pouvez réutiliser son `.env` (il doit aussi contenir
`LIVE_SERVER`) :

```bash
python main.py --config chemin/vers/live-server/.env --headless
```

Priorité : options `--server` / `--api-key`, puis variables d'environnement,
puis fichier `--config` (ou `translate/.env` par défaut, quel que soit le dossier
de lancement). Une URL ou une clé manquante bloque le lancement au lieu de
désactiver silencieusement la diffusion. Utilisez l'adresse réseau du serveur
à distance : `localhost` désigne toujours la machine qui exécute Translate.

Translate envoie aussi le texte français reconnu. Live Server l'utilise pour
traduire la phrase finale avec son modèle local. Avec Whisper, une seconde passe
ne concerne que la phrase finale afin de récupérer ce texte français ; elle peut
ajouter un court délai avant la correction serveur, sans ralentir les lignes en
cours.

Le mode Vosk fonctionne sur CPU. Pour Whisper sans configuration CUDA :
`python main.py --headless --engine whisper --whisper-device cpu`.
Le modèle Whisper est téléchargé au premier lancement. Le GPU Linux demande
une installation CUDA/cuDNN compatible avec votre version de CTranslate2.

### Windows / options communes

```bash
python main.py                     # auto : Whisper GPU si disponible
python main.py --engine vosk       # Vosk + traduction réseau
python main.py --engine google     # reconnaissance par phrase
python main.py --engine whisper    # 100% hors-ligne
python main.py --engine whisper --gpu-index 1  # choisir le second GPU
```

## Configuration par `.env`

Tous les réglages se font dans `translate/.env` (copie de `.env.example`, ignoré
par Git), donc `python main.py` suffit. Priorité : options `--xxx` > variables
d'environnement > fichier.

| Variable | Rôle |
|---|---|
| `LIVE_SERVER`, `LIVE_API_KEY` | Live Server et clé d'envoi (vides = pas de diffusion) |
| `TRANSLATE_ENGINE` | `auto`, `vosk`, `google` ou `whisper` |
| `TRANSLATE_LANGUAGE` | Langue parlée (`fr`) |
| `TRANSLATE_MIC_DEVICE` | Index du micro (`--list-devices`), vide = micro par défaut |
| `TRANSLATE_THRESHOLD` | Seuil de détection de la voix (google/whisper) |
| `WHISPER_MODEL` | `tiny` … `large-v3` (plus gros = plus précis) |
| `WHISPER_DEVICE`, `WHISPER_GPU_INDEX`, `WHISPER_BEAM_FINAL` | Calcul et précision de la finale |
| `VOSK_MODEL` | Dossier du modèle Vosk |
| `TRANSLATE_FONT_SIZE`, `TRANSLATE_HEADLESS` | Affichage |

## Choisir le micro

`python main.py --list-devices` liste les entrées ; `>` marque celle utilisée par
défaut. Renseignez l'index dans `TRANSLATE_MIC_DEVICE`. Préférez les entrées
**MME** ou **DirectSound** : WASAPI et WDM-KS refusent souvent les 16 kHz. Le micro
d'un casque sans fil est souvent bruité.

## Phrases parasites (Whisper)

Whisper a été entraîné sur des sous-titres : sur du bruit il invente des phrases
(« Subtitle … », « Thanks for watching »). Translate limite le problème :

- une transcription ne démarre qu'après au moins 300 ms de voix réelle ;
- les segments jugés sans parole ou peu fiables sont écartés, ainsi que les
  phrases-types d'hallucination ;
- avec un serveur, le français transcrit sert de référence : s'il est vide,
  rien n'est affiché ni envoyé.

Si cela persiste, augmentez `TRANSLATE_THRESHOLD` (ex. 0.03) ou changez de micro.

## Modèle Vosk

Le modèle français est dans `models/vosk-model-small-fr-0.22/`. S'il manque,
téléchargez-le une fois :

```bash
python -c "import urllib.request,zipfile,os; os.makedirs('models',exist_ok=True); urllib.request.urlretrieve('https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip','models/fr.zip'); zipfile.ZipFile('models/fr.zip').extractall('models')"
```

Pour plus de précision, un modèle plus gros existe
(`vosk-model-fr-0.22`, ~1,5 Go) : téléchargez-le et pointez dessus avec
`--vosk-model models/vosk-model-fr-0.22`.

## Options utiles

```bash
python main.py --list-devices      # liste les micros
python main.py --device 1          # choisir un micro précis
python main.py --font-size 52      # police plus grande
```

## Commandes à l'écran

- **Glisser la poignée** (petit carré rouge en haut à gauche) : déplacer les sous-titres
- **Double-clic sur la poignée** ou **Échap** : quitter
- **Molette** (au-dessus du texte) : taille de police

## Conseils pour une présentation

- Sous Windows, le fond est transparent : posez la fenêtre par-dessus votre PowerPoint/PDF,
  vous continuez à cliquer dessus normalement (le clic traverse).
- En mode `vosk`, la légende jaune pâle se précise en direct, puis passe en
  blanc quand la phrase est terminée.
- La traduction d'un texte qui grandit peut se réécrire (l'anglais réordonne
  les mots) : c'est normal pour des sous-titres traduits en direct.
