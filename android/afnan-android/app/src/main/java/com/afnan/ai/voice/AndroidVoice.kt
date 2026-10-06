package com.afnan.ai.voice

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

/**
 * SpeechRecognizer-based voice input. Defaults to Urdu (Pakistan);
 * callers pass any BCP-47 tag (e.g. "ur-PK", "en-US").
 */
class AndroidVoiceInput(private val context: Context) : VoiceInput {

    private var recognizer: SpeechRecognizer? = null
    private var active = false

    override fun isAvailable(): Boolean =
        SpeechRecognizer.isRecognitionAvailable(context)

    override fun startListening(
        languageTag: String,
        onResult: (String) -> Unit,
        onError: (String) -> Unit,
    ) {
        if (!isAvailable()) {
            onError("Speech recognition is not available on this device")
            return
        }
        stopListening()
        val listener = object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                active = false
            }

            override fun onError(error: Int) {
                active = false
                // ERROR_NO_MATCH / ERROR_SPEECH_TIMEOUT are routine;
                // surface them plainly so the UI can show a hint.
                onError(
                    when (error) {
                        SpeechRecognizer.ERROR_NO_MATCH -> "Didn't catch that — try again"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech heard"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS ->
                            "Microphone permission is required"

                        else -> "Voice input error ($error)"
                    },
                )
            }

            override fun onResults(results: Bundle) {
                active = false
                val text = results
                    .getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull()
                    .orEmpty()
                onResult(text)
            }

            override fun onPartialResults(partialResults: Bundle) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
        recognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
            setRecognitionListener(listener)
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM,
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageTag)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        active = true
        recognizer?.startListening(intent)
    }

    override fun stopListening() {
        if (active) {
            runCatching { recognizer?.stopListening() }
            active = false
        }
        runCatching { recognizer?.destroy() }
        recognizer = null
    }
}

/**
 * TextToSpeech output. Prefers the requested locale (ur-PK default);
 * falls back to the engine default when the voice data is missing.
 */
class AndroidVoiceOutput(private val context: Context) : VoiceOutput {

    @Volatile
    private var tts: TextToSpeech? = null

    @Volatile
    private var ready = false

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ready = true
            }
        }
    }

    override fun speak(text: String, languageTag: String, onDone: () -> Unit) {
        val engine = tts
        if (engine == null || !ready || text.isBlank()) {
            onDone()
            return
        }
        val locale = Locale.forLanguageTag(languageTag)
        val availability = engine.setLanguage(locale)
        if (availability == TextToSpeech.LANG_MISSING_DATA ||
            availability == TextToSpeech.LANG_NOT_SUPPORTED
        ) {
            // Fall back to the engine default voice rather than failing.
            engine.language = Locale.getDefault()
        }
        engine.setOnUtteranceProgressListener(
            object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) = onDone()
                override fun onError(utteranceId: String?) = onDone()
            },
        )
        engine.speak(
            text,
            TextToSpeech.QUEUE_FLUSH,
            null,
            "afnan-${System.currentTimeMillis()}",
        )
    }

    override fun stop() {
        runCatching { tts?.stop() }
    }

    fun shutdown() {
        runCatching { tts?.shutdown() }
        tts = null
        ready = false
    }
}
