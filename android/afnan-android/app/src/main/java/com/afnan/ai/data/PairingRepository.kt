package com.afnan.ai.data

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Device pairing against the `/v1/pair/request` and `/v1/pair/redeem`
 * bootstrap endpoints.
 *
 * Mirrors `afnan_ai/control/client/pairing.py` (PairingClient):
 * the 6-digit code lives in memory only and is never logged.
 */
class PairingRepository(
    private val client: ControlPlaneClient,
) {
    /**
     * Start pairing. Returns `Pair(pairing_id, code)`.
     *
     * Show [Pair.code] to the user exactly once; the server stores
     * only its hash. The owner must approve the request out-of-band
     * before [redeem] can succeed.
     */
    suspend fun requestPairing(
        deviceId: String,
        deviceName: String,
    ): Pair<String, String> {
        if (deviceId.isBlank()) {
            throw ProtocolException(
                "device_id is required",
                "invalid_metadata",
            )
        }
        val body = buildJsonObject {
            put("device_id", deviceId.trim())
            put("device_name", deviceName.take(120))
            put("platform", "android")
            put("client_version", CLIENT_VERSION)
        }
        val response = client.post("/v1/pair/request", body)
        val decoded = ControlJson.decodeFromJsonElement(
            PairingResponse.serializer(), response
        )
        if (decoded.pairingId.isEmpty() || decoded.code.isEmpty()) {
            throw ProtocolException(
                "pairing request returned no pairing_id/code",
                "pairing_failed",
            )
        }
        return decoded.pairingId to decoded.code
    }

    /**
     * Redeem an approved pairing code.
     *
     * Returns device id, session id, bearer token, capabilities and
     * expiry. The code is dropped as soon as the request is built.
     */
    suspend fun redeem(
        pairingId: String,
        code: String,
    ): RedeemResponse {
        val body = buildJsonObject {
            put("pairing_id", pairingId)
            // The code is only ever placed in this request body.
            put("code", code)
        }
        val response = client.post("/v1/pair/redeem", body)
        val decoded = ControlJson.decodeFromJsonElement(
            RedeemResponse.serializer(), response
        )
        if (decoded.token.isEmpty()) {
            throw ProtocolException(
                "pairing redeem returned no token",
                "pairing_failed",
            )
        }
        return decoded
    }

    companion object {
        /** Must match the app's versionName. */
        const val CLIENT_VERSION = "1.0.0"
    }
}
