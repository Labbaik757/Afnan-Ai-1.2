package com.afnan.ai.voice

/**
 * Speech-to-text input. Implementations must request RECORD_AUDIO
 * before starting; the UI handles the permission flow.
 */
interface VoiceInput {
    fun startListening(
        languageTag: String,
        onResult: (String) -> Unit,
        onError: (String) -> Unit,
    )

    fun stopListening()
    fun isAvailable(): Boolean
}

/** Text-to-speech output. */
interface VoiceOutput {
    fun speak(text: String, languageTag: String, onDone: () -> Unit = {})
    fun stop()
}
