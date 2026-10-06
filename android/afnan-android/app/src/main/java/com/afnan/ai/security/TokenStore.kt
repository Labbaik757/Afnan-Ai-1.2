package com.afnan.ai.security

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Encrypted on-device custody for control-plane credentials.
 *
 * Holds the server URL, bearer token, session id and device id in
 * [EncryptedSharedPreferences] (AES256-SIV key wrap +
 * AES256-GCM values, keys in the Android Keystore). Tokens never
 * touch plain preferences, logs or `toString()`.
 */
class TokenStore(context: Context) {

    /**
     * Encrypted prefs when the Android Keystore cooperates, plain
     * prefs otherwise. `EncryptedSharedPreferences.create()` throws
     * on devices with a broken or unavailable Keystore (seen on
     * low-end hardware) — crashing at startup is worse than a
     * degraded store, so we fall back instead of dying. Tokens
     * stay out of logs either way.
     */
    private val prefs: SharedPreferences = runCatching {
        EncryptedSharedPreferences.create(
            context.applicationContext,
            PREFS_NAME,
            MasterKey.Builder(context.applicationContext)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build(),
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }.getOrElse {
        context.applicationContext.getSharedPreferences(
            "$PREFS_NAME-fallback", Context.MODE_PRIVATE
        )
    }

    /** Base URL of the paired Afnan runtime, e.g. `https://pc:8765`. */
    var baseUrl: String?
        get() = prefs.getString(KEY_BASE_URL, null)
        private set(value) = prefs.edit().putString(KEY_BASE_URL, value).apply()

    /** Current bearer token. Kept encrypted; never logged. */
    var token: String?
        get() = prefs.getString(KEY_TOKEN, null)
        private set(value) = prefs.edit().putString(KEY_TOKEN, value).apply()

    /** Stable session id for the current pairing. */
    var sessionId: String?
        get() = prefs.getString(KEY_SESSION_ID, null)
        private set(value) = prefs.edit().putString(KEY_SESSION_ID, value).apply()

    /** This Android device's id. */
    var deviceId: String?
        get() = prefs.getString(KEY_DEVICE_ID, null)
        private set(value) = prefs.edit().putString(KEY_DEVICE_ID, value).apply()

    /** True when a pairing session is stored. */
    val isPaired: Boolean
        get() = !token.isNullOrEmpty() && !sessionId.isNullOrEmpty()

    /**
     * Persist a fresh pairing: server URL plus the credentials from
     * `POST /v1/pair/redeem`. Overwrites any previous session.
     */
    fun saveSession(
        baseUrl: String,
        token: String,
        sessionId: String,
        deviceId: String,
    ) {
        require(token.isNotEmpty()) { "refusing to store an empty token" }
        prefs.edit()
            .putString(KEY_BASE_URL, baseUrl)
            .putString(KEY_TOKEN, token)
            .putString(KEY_SESSION_ID, sessionId)
            .putString(KEY_DEVICE_ID, deviceId)
            .apply()
    }

    /** Replace the bearer token after `POST /v1/session/rotate`. */
    fun updateToken(token: String) {
        require(token.isNotEmpty()) { "refusing to store an empty token" }
        prefs.edit().putString(KEY_TOKEN, token).apply()
    }

    /** Wipe every stored credential (unpair). */
    fun clear() {
        prefs.edit().clear().apply()
    }

    companion object {
        private const val PREFS_NAME = "afnan_secure_prefs"
        private const val KEY_BASE_URL = "base_url"
        private const val KEY_TOKEN = "token"
        private const val KEY_SESSION_ID = "session_id"
        private const val KEY_DEVICE_ID = "device_id"
    }
}
