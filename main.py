"""
Traduction vocale en temps reel : francais parle -> anglais ecrit.

Sous-titres facon "film" : texte centre en bas de l'ecran, blanc a contour
noir, fond transparent (les clics passent a travers vers votre presentation).
La legende se construit mot a mot pendant que vous parlez.

Trois moteurs (selection automatique de Whisper GPU par defaut) :
  --engine vosk             : reconnaissance en STREAMING (Vosk, local) qui
      emet des resultats partiels mot a mot ; chaque partiel est traduit en
      continu (Google Translate, sans cle). -> legende vivante en temps reel.
      Necessite internet pour la traduction uniquement.
  --engine google           : reconnaissance Google par phrase (precis mais
      pas de mot a mot). Necessite internet.
  --engine whisper          : faster-whisper 100% local/hors-ligne.

Raccourcis :
  - Glisser la poignee (coin haut-gauche) : deplacer les sous-titres
  - Double-clic sur la poignee, ou Echap  : quitter
  - Molette (au-dessus du texte)          : taille de police
"""

import argparse
import faulthandler
import json
import os
import queue
import re
import sys
import threading
import time

import numpy as np

from configuration import app_dir, load_settings, publish_url


# --------------------------------------------------------------------------
# Reglages audio / VAD
# --------------------------------------------------------------------------
if os.environ.get("TRANSLATE_DEBUG"):
    # Diagnostic : affiche ou le programme est bloque apres 20 s.
    faulthandler.dump_traceback_later(20, exit=False)

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000

SILENCE_MS = 600             # (google/whisper) silence qui termine une phrase
MIN_SPEECH_MS = 300
MAX_SPEECH_MS = 12000
PREROLL_MS = 300
INTERIM_REFRESH_MS = 300

INTERIM_TR_MS = 800          # (vosk) evite de saturer les services de traduction gratuits
LINGER_SECONDS = 4.0         # duree d'affichage d'une phrase apres la fin

SCRIPT_DIR = str(app_dir())
DEFAULT_VOSK_MODEL = os.path.join(SCRIPT_DIR, "models", "vosk-model-small-fr-0.22")

_CUDA_DLL_HANDLES = []


# Whisper a ete entraine sur des sous-titres : sur du bruit ou du silence il
# invente volontiers ces phrases-types.
_HALLUCINATION_RE = re.compile(
    r"subtitle|sous-?titr|amara\.org|thanks? for watching|thank you for watching|"
    r"merci d'avoir regard|abonnez-vous|please subscribe|www\.|\.com|"
    r"^[\s\-–.,!?♪]*$",
    re.IGNORECASE,
)
NO_SPEECH_MAX = 0.6


def clean_segments(segments):
    """Assemble les segments Whisper en ecartant ceux qui sont du bruit."""
    parts = []
    for seg in segments:
        text = seg.text.strip()
        if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_MAX:
            continue
        if getattr(seg, "avg_logprob", 0.0) < -1.0:
            continue
        if _HALLUCINATION_RE.search(text):
            continue
        parts.append(text)
    return " ".join(parts).strip()


def configure_cuda_dlls():
    """Expose les DLL NVIDIA installees par pip a CTranslate2 sous Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    roots = [os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")]
    if getattr(sys, "frozen", False):
        roots.insert(0, os.path.join(getattr(sys, "_MEIPASS", ""), "nvidia"))
    for nvidia_dir in roots:
        for component in ("cublas", "cudnn", "cuda_nvrtc"):
            bin_dir = os.path.join(nvidia_dir, component, "bin")
            if os.path.isdir(bin_dir):
                _CUDA_DLL_HANDLES.append(os.add_dll_directory(bin_dir))
                if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

# --------------------------------------------------------------------------
# Reglages affichage (sous-titres facon film)
# --------------------------------------------------------------------------
TRANSPARENT_KEY = "#0b0b0b"       # couleur rendue transparente par Windows
OUTLINE = "#000000"
TEXT_FINAL = "#ffffff"            # blanc : phrase terminee
TEXT_LIVE = "#fff2a8"             # jaune pale : phrase en cours (mot a mot)
OUTLINE_R = 2                     # epaisseur du contour (px)


class SubtitleApp:
    """Sous-titres transparents facon film, texte contoure centre en bas."""

    def __init__(self, text_queue, font_size=40):
        global tk
        import tkinter as tk

        self.text_queue = text_queue
        self.font_size = font_size

        self.root = tk.Tk()
        self.root.title("Traduction FR -> EN")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            pass  # non supporte hors Windows -> fond opaque
        self.root.configure(bg=TRANSPARENT_KEY)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.width = self.screen_w
        self.height = int(self.screen_h * 0.32)
        self.root.geometry(f"{self.width}x{self.height}+0+{self.screen_h - self.height}")

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height,
                                bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.text = ""
        self.color = TEXT_FINAL
        self.last_final_utt = -1
        self.clear_at = None          # instant ou effacer la legende
        self._drag = {"x": 0, "y": 0}

        # Poignee opaque (deplacer / quitter) — cliquable meme en fond transparent.
        self.handle = self.canvas.create_rectangle(
            8, 8, 30, 30, fill="#e02424", outline="", tags="handle")
        self.canvas.tag_bind("handle", "<Button-1>", self._start_drag)
        self.canvas.tag_bind("handle", "<B1-Motion>", self._on_drag)
        self.canvas.tag_bind("handle", "<Double-Button-1>", lambda e: self.stop())

        self.root.bind("<Escape>", lambda e: self.stop())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.root.after(50, self.root.focus_force)

        self._running = True
        self._pulse = True
        self.root.after(60, self._poll_text)
        self.root.after(200, self._tick)
        self._draw()

    # --- dessin des sous-titres avec contour ---
    def _draw(self):
        self.canvas.delete("caption")
        if not self.text:
            return
        x = self.width // 2
        y = self.height - 28
        wrap = int(self.screen_w * 0.82)
        font = ("Arial", self.font_size, "bold")
        r = OUTLINE_R
        offsets = [(-r, -r), (-r, 0), (-r, r), (0, -r), (0, r), (r, -r), (r, 0), (r, r)]
        for dx, dy in offsets:
            self.canvas.create_text(x + dx, y + dy, text=self.text, fill=OUTLINE,
                                    font=font, width=wrap, anchor="s",
                                    justify="center", tags="caption")
        self.canvas.create_text(x, y, text=self.text, fill=self.color,
                                font=font, width=wrap, anchor="s",
                                justify="center", tags="caption")

    # --- interactions ---
    def _start_drag(self, event):
        self._drag["x"] = event.x_root - self.root.winfo_x()
        self._drag["y"] = event.y_root - self.root.winfo_y()

    def _on_drag(self, event):
        x = event.x_root - self._drag["x"]
        y = event.y_root - self._drag["y"]
        self.root.geometry(f"+{x}+{y}")

    def _on_wheel(self, event):
        up = getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0
        self.font_size = max(16, min(90, self.font_size + (3 if up else -3)))
        self._draw()

    # --- reception des textes ---
    def _poll_text(self):
        if not self._running:
            return
        changed = False
        try:
            while True:
                item = self.text_queue.get_nowait()
                kind, utt_id, text = item[:3]
                if kind == "interim":
                    if utt_id > self.last_final_utt:
                        self.text, self.color, self.clear_at = text, TEXT_LIVE, None
                        changed = True
                elif kind == "final":
                    self.last_final_utt = max(self.last_final_utt, utt_id)
                    if text:
                        self.text, self.color = text, TEXT_FINAL
                        self.clear_at = time.monotonic() + LINGER_SECONDS
                        changed = True
                elif kind == "status":
                    self.text, self.color, self.clear_at = text, TEXT_LIVE, None
                    changed = True
        except queue.Empty:
            pass
        if changed:
            self._draw()
        self.root.after(60, self._poll_text)

    def _tick(self):
        if not self._running:
            return
        # Effacement de la phrase apres un temps de latence.
        if self.clear_at is not None and time.monotonic() >= self.clear_at:
            self.text, self.clear_at = "", None
            self._draw()
        # Clignotement de la poignee.
        self._pulse = not self._pulse
        self.canvas.itemconfig(self.handle, fill="#e02424" if self._pulse else "#3a0f0f")
        self.root.after(400, self._tick)

    def stop(self):
        self._running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


# ==========================================================================
# Traducteur throttle : finales prioritaires, partiels ecrases si en retard
# ==========================================================================
class ResilientTranslator:
    """Google en priorite, avec MyMemory en secours et cache local."""

    GOOGLE_RETRY_SECONDS = 60.0
    ERROR_LOG_SECONDS = 15.0
    CACHE_MAX = 256

    def __init__(self, language):
        from deep_translator import GoogleTranslator, MyMemoryTranslator

        self._google = GoogleTranslator(source=language, target="en")
        self._google_retry_at = 0.0
        self._last_error_log = 0.0
        self._cache = {}

        # MyMemory attend des codes regionaux, contrairement a Google.
        source = {
            "fr": "fr-FR", "en": "en-GB", "de": "de-DE",
            "es": "es-ES", "it": "it-IT", "pt": "pt-PT",
        }.get(language.lower())
        self._fallback = (
            MyMemoryTranslator(source=source, target="en-GB") if source else None
        )

    def _remember(self, source, translated):
        if not translated:
            return translated
        self._cache[source] = translated
        if len(self._cache) > self.CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        return translated

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            return ""
        if text in self._cache:
            return self._cache[text]

        now = time.monotonic()
        google_error = None
        if now >= self._google_retry_at:
            try:
                return self._remember(text, self._google.translate(text) or "")
            except Exception as exc:  # noqa: BLE001
                google_error = exc
                self._google_retry_at = now + self.GOOGLE_RETRY_SECONDS

        if self._fallback is not None:
            try:
                translated = self._fallback.translate(text) or ""
                if google_error is not None:
                    print(
                        "[traduction] Google indisponible; bascule temporaire "
                        "vers MyMemory.",
                        file=sys.stderr,
                    )
                return self._remember(text, translated)
            except Exception as fallback_error:  # noqa: BLE001
                if now - self._last_error_log >= self.ERROR_LOG_SECONDS:
                    detail = google_error or fallback_error
                    print(f"[erreur traduction] {detail}", file=sys.stderr)
                    self._last_error_log = now
        elif google_error is not None and now - self._last_error_log >= self.ERROR_LOG_SECONDS:
            print(f"[erreur traduction] {google_error}", file=sys.stderr)
            self._last_error_log = now

        # Une chaine vide signale l'echec sans faire passer le francais pour
        # une traduction anglaise.
        return ""


class ThrottledTranslator:
    """Traduit FR -> EN en tache de fond. Les finales passent avant les partiels."""

    def __init__(self, language, text_queue, stop_event):
        self._translator = ResilientTranslator(language)
        self.text_queue = text_queue
        self.stop_event = stop_event
        self.final_q = queue.Queue()
        self._slot = [None]
        self._lock = threading.Lock()
        self._last_by_utt = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def _translate(self, fr):
        return self._translator.translate(fr)

    def submit_final(self, utt_id, fr):
        # Ne pas laisser un ancien partiel etre publie apres sa phrase finale.
        with self._lock:
            if self._slot[0] is not None and self._slot[0][0] == utt_id:
                self._slot[0] = None
        self.final_q.put((utt_id, fr))

    def submit_interim(self, utt_id, fr):
        with self._lock:
            self._slot[0] = (utt_id, fr)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                utt_id, fr = self.final_q.get_nowait()
                en = self._translate(fr)
                last_interim = self._last_by_utt.pop(utt_id, "")
                en = en or last_interim
                self.text_queue.put(("final", utt_id, en, fr))
                if en:
                    print(f"EN> {en}")
                continue
            except queue.Empty:
                pass
            with self._lock:
                job = self._slot[0]
                self._slot[0] = None
            if job is None:
                time.sleep(0.01)
                continue
            utt_id, fr = job
            en = self._translate(fr)
            if en:
                self._last_by_utt[utt_id] = en
            self.text_queue.put(("interim", utt_id, en, fr))


# ==========================================================================
# Moteur Vosk : reconnaissance en streaming, mot a mot
# ==========================================================================
def run_vosk_engine(audio_queue, text_queue, language, stop_event, model_path):
    from vosk import Model, KaldiRecognizer

    if not os.path.isdir(model_path):
        print(f"[erreur] Modele Vosk introuvable : {model_path}", file=sys.stderr)
        print("Telechargez-le : voir README (models/vosk-model-small-fr-0.22).", file=sys.stderr)
        stop_event.set()
        return

    print("Chargement du modele Vosk...")
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)
    print("Modele pret. Parlez ! (Echap pour quitter)\n")

    translator = ThrottledTranslator(language, text_queue, stop_event)

    utt_id = 0
    last_partial = ""
    last_tr = 0.0

    while not stop_event.is_set():
        try:
            frame = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        cur = utt_id + 1

        if rec.AcceptWaveform(pcm):
            fr = json.loads(rec.Result()).get("text", "").strip()
            if fr:
                translator.submit_final(cur, fr)
                print(f"FR> {fr}")
                utt_id = cur
            last_partial = ""
        else:
            fr = json.loads(rec.PartialResult()).get("partial", "").strip()
            now = time.monotonic()
            if fr and fr != last_partial and (now - last_tr) >= INTERIM_TR_MS / 1000:
                last_partial = fr
                last_tr = now
                translator.submit_interim(cur, fr)


# ==========================================================================
# VAD par energie (moteurs google / whisper)
# ==========================================================================
def vad_segments(audio_queue, threshold, stop_event, on_start, on_interim, on_final):
    from collections import deque
    silence_needed = SILENCE_MS // FRAME_MS
    min_speech = MIN_SPEECH_MS // FRAME_MS
    max_speech = MAX_SPEECH_MS // FRAME_MS
    preroll_len = PREROLL_MS // FRAME_MS
    refresh_frames = INTERIM_REFRESH_MS // FRAME_MS

    speech = []
    preroll = deque(maxlen=preroll_len)
    in_speech = False
    silence_count = 0
    utt_id = 0
    frames_since_interim = 0

    voiced = 0

    def finalize():
        nonlocal speech, voiced
        # Les trames de silence et de pre-roll ne comptent pas : un simple
        # bruit ne doit pas declencher une transcription (hallucinations).
        if voiced >= min_speech:
            on_final(utt_id, np.concatenate(speech).astype(np.float32))
        speech = []
        voiced = 0

    while not stop_event.is_set():
        try:
            frame = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > threshold:
            if not in_speech:
                in_speech = True
                utt_id += 1
                speech = list(preroll)
                voiced = 0
                frames_since_interim = 0
                on_start(utt_id)
            speech.append(frame)
            voiced += 1
            silence_count = 0
            frames_since_interim += 1
            if on_interim and voiced >= min_speech and frames_since_interim >= refresh_frames:
                on_interim(utt_id, np.concatenate(speech).astype(np.float32))
                frames_since_interim = 0
            if len(speech) >= max_speech:
                finalize()
                in_speech = False
        else:
            if in_speech:
                speech.append(frame)
                silence_count += 1
                if silence_count >= silence_needed:
                    finalize()
                    in_speech = False
                    silence_count = 0
            else:
                preroll.append(frame)


def run_google_engine(audio_queue, text_queue, threshold, language, stop_event):
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    translator = ResilientTranslator(language)
    sr_lang = {"fr": "fr-FR"}.get(language, language)
    jobs = queue.Queue()

    def worker():
        while not stop_event.is_set():
            try:
                utt_id, audio = jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            try:
                fr = recognizer.recognize_google(sr.AudioData(pcm, SAMPLE_RATE, 2), language=sr_lang)
            except sr.UnknownValueError:
                text_queue.put(("final", utt_id, ""))
                continue
            except sr.RequestError as exc:
                print(f"[erreur reseau Google] {exc}", file=sys.stderr)
                text_queue.put(("final", utt_id, ""))
                continue
            en = translator.translate(fr)
            print(f"FR> {fr}\nEN> {en}\n")
            text_queue.put(("final", utt_id, en or "", fr))

    threading.Thread(target=worker, daemon=True).start()
    vad_segments(audio_queue, threshold, stop_event,
                 on_start=lambda utt: text_queue.put(("status", utt, "...")),
                 on_interim=None,
                 on_final=lambda utt, audio: jobs.put((utt, audio)))


def run_whisper_engine(audio_queue, text_queue, threshold, language,
                      model_name, beam_final, stop_event, whisper_device,
                      gpu_index):
    configure_cuda_dlls()
    from faster_whisper import WhisperModel

    if whisper_device == "auto":
        import ctranslate2
        whisper_device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    compute_type = "float16" if whisper_device == "cuda" else "int8"
    device_label = f"CUDA:{gpu_index}" if whisper_device == "cuda" else "CPU"
    print(f"Chargement du modele Whisper '{model_name}' ({compute_type}, {device_label})...")
    model = WhisperModel(
        model_name,
        device=whisper_device,
        device_index=gpu_index if whisper_device == "cuda" else 0,
        compute_type=compute_type,
    )
    # CTranslate2 initialise une partie du backend au premier appel. Faire ce
    # travail avant l'ouverture du micro evite de ralentir la premiere phrase.
    warmup, _ = model.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        language=language,
        task="translate",
        beam_size=1,
    )
    list(warmup)
    print("Modele pret.\n")

    def translate(audio, beam_size, use_vad):
        segments, _ = model.transcribe(
            audio, language=language, task="translate", beam_size=beam_size,
            temperature=0.0, vad_filter=use_vad, condition_on_previous_text=False,
            no_speech_threshold=0.6, log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4)
        return clean_segments(segments)

    # Le resultat anglais local reste immediat. Lorsqu'un serveur est configure,
    # une seconde passe sur les seules phrases finales fournit le francais brut
    # au modele serveur, sans ralentir les sous-titres provisoires.
    publish_source = getattr(text_queue, "publisher", None) is not None

    def transcribe_source(audio):
        segments, _ = model.transcribe(
            audio, language=language, task="transcribe", beam_size=beam_final,
            temperature=0.0, vad_filter=True, condition_on_previous_text=False,
            no_speech_threshold=0.6, log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4)
        return clean_segments(segments)

    final_q = queue.Queue()
    slot = [None]
    lock = threading.Lock()

    def infer():
        while not stop_event.is_set():
            try:
                utt_id, audio = final_q.get_nowait()
                if publish_source:
                    # Le francais sert de reference : s'il est vide, c'etait du
                    # bruit et rien n'est affiche ni diffuse.
                    source = transcribe_source(audio)
                    text = translate(audio, beam_final, True) if source else ""
                else:
                    source = None
                    text = translate(audio, beam_final, True)
                text_queue.put(("final", utt_id, text, source))
                continue
            except queue.Empty:
                pass
            with lock:
                job = slot[0]
                slot[0] = None
            if job is None:
                time.sleep(0.01)
                continue
            utt_id, audio = job
            text_queue.put(("interim", utt_id, translate(audio, 1, False)))

    threading.Thread(target=infer, daemon=True).start()

    def set_interim(utt_id, audio):
        with lock:
            slot[0] = (utt_id, audio)

    vad_segments(audio_queue, threshold, stop_event,
                 on_start=lambda utt: None,
                 on_interim=set_interim,
                 on_final=lambda utt, audio: final_q.put((utt, audio)))


# ==========================================================================
# Diffusion vers le serveur distant (optionnelle)
# ==========================================================================
def _to_publish_ws(base):
    """URL de base http(s)://host -> ws(s)://host/ws/publish."""
    return publish_url(base)


class RemotePublisher:
    """Envoie interim/final au serveur via WebSocket, avec reconnexion auto.

    Les finales sont prioritaires ; seul le dernier interim en attente est garde
    (les interims perimes sont abandonnes pour ne pas saturer le reseau).
    """

    def __init__(self, server_base, api_key, stop_event):
        self.url = _to_publish_ws(server_base)
        self.api_key = api_key
        self.stop_event = stop_event
        self.final_q = queue.Queue()
        self._slot = [None]
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def send(self, kind, text, source=None):
        if kind == "final":
            with self._lock:
                self._slot[0] = None
            self.final_q.put((text, source))
        elif kind == "interim":
            with self._lock:
                self._slot[0] = (text, source)

    def _next(self):
        try:
            return ("final", self.final_q.get_nowait())
        except queue.Empty:
            pass
        with self._lock:
            text = self._slot[0]
            self._slot[0] = None
        return ("interim", text) if text is not None else None

    def _run(self):
        from websockets.sync.client import connect
        pending = None
        while not self.stop_event.is_set():
            try:
                with connect(self.url, additional_headers={"x-api-key": self.api_key},
                             open_timeout=5, close_timeout=2,
                             ping_interval=20, ping_timeout=90) as ws:
                    print(f"[serveur] connecte -> {self.url}")
                    while not self.stop_event.is_set():
                        job = pending or self._next()
                        if job is None:
                            time.sleep(0.02)
                            continue
                        kind, (text, source) = job
                        pending = job if kind == "final" else None
                        payload = {"kind": kind, "text": text}
                        if source is not None:
                            payload["source"] = source
                        ws.send(json.dumps(payload, ensure_ascii=False))
                        pending = None
            except Exception as exc:  # noqa: BLE001
                print(f"[serveur] deconnecte ({type(exc).__name__}); verifier le serveur et LIVE_API_KEY ; reconnexion dans 2s...", file=sys.stderr)
                self.stop_event.wait(2)


class Tee:
    """File compatible .put qui alimente l'UI locale ET le serveur distant."""

    def __init__(self, ui_queue, publisher):
        self.ui_queue = ui_queue
        self.publisher = publisher

    def put(self, item):
        if self.ui_queue is not None:
            self.ui_queue.put(item)
        kind, _utt, text, *rest = item
        source = rest[0] if rest else None
        if self.publisher is not None and kind in ("interim", "final"):
            self.publisher.send(kind, text, source)
        if self.ui_queue is None and kind == "final" and text:
            print(f"EN> {text}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Traduction vocale FR -> EN en temps reel.")
    parser.add_argument("--engine", choices=["auto", "vosk", "google", "whisper"], default=None,
                        help="Moteur : auto (Whisper si GPU), vosk, google ou whisper.")
    parser.add_argument("--vosk-model", default=None, help="Dossier du modele Vosk.")
    parser.add_argument("--model", default=None, help="[whisper] tiny/base/small/medium/large-v3")
    parser.add_argument("--device", type=int, default=None, help="Index du micro (voir --list-devices).")
    parser.add_argument("--language", default=None, help="Langue parlee (defaut: fr)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="[google/whisper] Seuil de detection de la voix.")
    parser.add_argument("--beam-final", type=int, default=None, help="[whisper] Faisceau de la finale.")
    parser.add_argument("--whisper-device", choices=["auto", "cuda", "cpu"], default=None,
                        help="[whisper] Calcul sur GPU NVIDIA si disponible, sinon CPU.")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="[whisper] Index du GPU CUDA (voir nvidia-smi).")
    parser.add_argument("--font-size", type=int, default=None, help="Taille de police de depart.")
    parser.add_argument("--server", default=None,
                        help="URL du serveur de diffusion (ex: https://mon-serveur). Diffuse la traduction.")
    parser.add_argument("--api-key", default=None, help="Cle API pour pousser vers le serveur.")
    parser.add_argument("--config", help="Fichier .env (defaut : translate/.env).")
    parser.add_argument("--headless", action="store_true",
                        help="Sans fenetre, pour Linux sans bureau. Arret par Ctrl+C.")
    parser.add_argument("--list-devices", action="store_true", help="Liste les micros et quitte.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        args.server = args.server if args.server is not None else settings['LIVE_SERVER']
        args.api_key = args.api_key if args.api_key is not None else settings['LIVE_API_KEY']

        def pick(value, name, default, cast=str):
            if value is not None:
                return value
            raw = (settings.get(name) or '').strip()
            if not raw:
                return default
            try:
                return cast(raw)
            except ValueError:
                raise ValueError(f'{name} invalide : {raw!r}') from None

        def to_bool(raw):
            return raw.lower() in ('1', 'true', 'yes', 'on')

        args.engine = pick(args.engine, 'TRANSLATE_ENGINE', 'auto')
        if args.engine not in ('auto', 'vosk', 'google', 'whisper'):
            raise ValueError('TRANSLATE_ENGINE doit valoir auto, vosk, google ou whisper.')
        args.vosk_model = pick(args.vosk_model, 'VOSK_MODEL', DEFAULT_VOSK_MODEL)
        args.model = pick(args.model, 'WHISPER_MODEL', 'small')
        args.language = pick(args.language, 'TRANSLATE_LANGUAGE', 'fr')
        args.threshold = pick(args.threshold, 'TRANSLATE_THRESHOLD', 0.015, float)
        args.beam_final = pick(args.beam_final, 'WHISPER_BEAM_FINAL', 5, int)
        args.whisper_device = pick(args.whisper_device, 'WHISPER_DEVICE', 'auto')
        if args.whisper_device not in ('auto', 'cuda', 'cpu'):
            raise ValueError('WHISPER_DEVICE doit valoir auto, cuda ou cpu.')
        args.gpu_index = pick(args.gpu_index, 'WHISPER_GPU_INDEX', 0, int)
        args.font_size = pick(args.font_size, 'TRANSLATE_FONT_SIZE', 40, int)
        args.device = pick(args.device, 'TRANSLATE_MIC_DEVICE', None, int)
        if not args.headless:
            args.headless = pick(None, 'TRANSLATE_HEADLESS', False, to_bool)
        if not args.list_devices:
            if bool(args.server) != bool(args.api_key):
                raise ValueError('LIVE_SERVER et LIVE_API_KEY (ou --server et --api-key) sont requis ensemble.')
            if args.server:
                publish_url(args.server)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        args.headless = True
    return args


def main():
    args = parse_args()
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise SystemExit(
            'Audio indisponible : installez requirements.txt et, sous Ubuntu/Debian, '
            'libportaudio2 (sudo apt-get install libportaudio2).'
        ) from exc

    if args.list_devices:
        print(sd.query_devices())
        return

    audio_queue = queue.Queue()
    ui_queue = None if args.headless else queue.Queue()
    stop_event = threading.Event()
    app = None
    if not args.headless:
        try:
            app = SubtitleApp(ui_queue, font_size=args.font_size)
        except Exception as exc:
            raise SystemExit('Interface indisponible : utilisez --headless ou installez python3-tk avec un affichage X11.') from exc

    # Diffusion distante optionnelle : le Tee alimente l'UI locale + le serveur.
    publisher = None
    if args.server:
        publisher = RemotePublisher(args.server, args.api_key, stop_event)
        print(f"Diffusion activee vers {args.server}")
    text_queue = Tee(ui_queue, publisher)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    engine = args.engine
    if engine == "auto":
        try:
            configure_cuda_dlls()
            import ctranslate2
            engine = "whisper" if ctranslate2.get_cuda_device_count() else "vosk"
        except Exception:  # noqa: BLE001
            engine = "vosk"
        print(f"Moteur selectionne automatiquement : {engine}")

    if engine == "vosk":
        target = run_vosk_engine
        engine_args = (audio_queue, text_queue, args.language, stop_event, args.vosk_model)
    elif engine == "google":
        target = run_google_engine
        engine_args = (audio_queue, text_queue, args.threshold, args.language, stop_event)
    else:
        target = run_whisper_engine
        engine_args = (audio_queue, text_queue, args.threshold, args.language,
                       args.model, args.beam_final, stop_event,
                       args.whisper_device, args.gpu_index)

    def run_engine():
        try:
            target(*engine_args)
        except Exception as exc:
            print(f"[erreur moteur] {exc}", file=sys.stderr)
            stop_event.set()

    threading.Thread(target=run_engine, daemon=True).start()

    def check_engine():
        if stop_event.is_set():
            app.stop()
        else:
            app.root.after(200, check_engine)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, channels=1,
                            dtype="float32", device=args.device, callback=audio_callback):
            if app is None:
                print('Mode sans interface. Ctrl+C pour quitter.')
                while not stop_event.wait(0.2):
                    pass
            else:
                app.root.after(200, check_engine)
                app.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if app is not None:
            app.stop()
        if publisher is not None:
            publisher.thread.join(timeout=7)
    print("\nArret.")


if __name__ == "__main__":
    main()
