"""Voz offline con Piper. Import diferido: la app arranca aunque falte la voz."""

from __future__ import annotations

__all__ = ["SpeechEngine", "SpeechStatus"]


def __getattr__(name: str):  # noqa: D103 - import diferido
    if name in __all__:
        from senasperu.tts import speech

        return getattr(speech, name)
    raise AttributeError(f"El módulo 'senasperu.tts' no tiene el atributo '{name}'.")
