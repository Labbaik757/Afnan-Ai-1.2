package com.afnan.ai.data

import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

// ---------------------------------------------------------------------------
// Errors — mirrors afnan_ai/control/client/errors.py. Token values are never
// embedded in messages; see [redactTokenText].
// ---------------------------------------------------------------------------

/** Base error for control-plane protocol failures. */
open class ProtocolException(
    message: String = "",
    val code: String = "protocol_error",
    val status: Int? = null,
) : IOException(
    if (status != null) "[$code] HTTP $status: $message"
    else "[$code] $message"
)

/** The token was missing, expired, revoked or otherwise invalid (HTTP 401). */
class AuthenticationException(
    message: String = "authentication failed",
    code: String = "unauthorized",
) : ProtocolException(message, code, 401)

/** Authenticated but lacking the capability (HTTP 403). */
class AuthorizationException(
    message: String = "not authorized",
    code: String = "forbidden",
) : ProtocolException(message, code, 403)

/** The server is rate-limiting this client (HTTP 429). */
class RateLimitException(
    message: String = "rate limit exceeded",
    code: String = "rate_limited",
) : ProtocolException(message, code, 429)

/** The transport itself failed (network, timeout, TLS, refused). */
class ConnectionException(
    message: String = "connection failed",
    code: String = "connection_error",
) : ProtocolException(message, code, null)

private val BEARER_RE = Regex("Bearer\\s+[^\\s\"']+", RegexOption.IGNORE_CASE)
private val QUERY_TOKEN_RE =
    Regex("([?&]token=)[^&\\s\"']+", RegexOption.IGNORE_CASE)

/**
 * Redact bearer tokens and token query parameters from a string.
 * Applied to every error message and log line leaving this module.
 */
fun redactTokenText(text: String): String =
    QUERY_TOKEN_RE.replace(
        BEARER_RE.replace(text, "Bearer <redacted>"),
        "$1<redacted>",
    )

// ---------------------------------------------------------------------------
// HTTP client — mirrors afnan_ai/control/client/base.py (ControlClient).
// ---------------------------------------------------------------------------

/**
 * HTTP transport for the control-plane `/v1` API.
 *
 * Uses [HttpURLConnection] with JSON bodies, `Authorization: Bearer`
 * headers and typed error mapping. All blocking I/O runs on
 * [Dispatchers.IO]; callers use suspend functions from any dispatcher.
 *
 * @param baseUrl e.g. `http://127.0.0.1:8765` or `https://pc:8765`.
 *   A trailing slash is tolerated.
 * @param timeoutMs default connect/read timeout per request.
 */
class ControlPlaneClient(
    baseUrl: String,
    val timeoutMs: Int = 30_000,
) {
    val baseUrl: String = baseUrl.trimEnd('/')
    private val closed = AtomicBoolean(false)

    /** Mark the client closed; further requests are refused. */
    fun close() {
        closed.set(true)
    }

    private fun checkOpen() {
        if (closed.get()) {
            throw ConnectionException("client is closed", "client_closed")
        }
    }

    /**
     * Send one request and return the decoded JSON object body.
     *
     * @throws AuthenticationException on 401
     * @throws AuthorizationException on 403
     * @throws RateLimitException on 429
     * @throws ProtocolException on 400/404/409 and other HTTP errors
     * @throws ConnectionException on transport failures
     */
    suspend fun request(
        method: String,
        path: String,
        body: JsonObject? = null,
        token: String? = null,
        timeoutMs: Int = this.timeoutMs,
    ): JsonObject = withContext(Dispatchers.IO) {
        checkOpen()
        val url = URL(this@ControlPlaneClient.baseUrl + path)
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method.uppercase()
            connectTimeout = timeoutMs
            readTimeout = timeoutMs
            doInput = true
            setRequestProperty("Accept", "application/json")
            setRequestProperty("User-Agent", "AfnanAndroidClient/1.0")
            if (!token.isNullOrEmpty()) {
                // The header value is set directly on the connection
                // and never copied into an exception message.
                setRequestProperty("Authorization", "Bearer $token")
            }
            if (body != null) {
                doOutput = true
                setRequestProperty(
                    "Content-Type", "application/json; charset=utf-8"
                )
            }
        }
        try {
            if (body != null) {
                val bytes = ControlJson.encodeToString(
                    JsonObject.serializer(), body
                ).toByteArray(Charsets.UTF_8)
                connection.outputStream.use { it.write(bytes) }
            }
            val status = connection.responseCode
            val raw = readBody(connection, status)
            if (status in 200..299) {
                return@withContext parseObject(raw, method, path)
            }
            throw mapHttpError(status, raw, method, path)
        } catch (e: ProtocolException) {
            throw e
        } catch (e: SocketTimeoutException) {
            throw ConnectionException(
                "${method.uppercase()} $path: timed out",
                "timeout",
            )
        } catch (e: IOException) {
            throw ConnectionException(
                "${method.uppercase()} $path: ${e.javaClass.simpleName}",
                "transport_failure",
            )
        } finally {
            connection.disconnect()
        }
    }

    /** Convenience GET. */
    suspend fun get(
        path: String,
        token: String? = null,
        timeoutMs: Int = this.timeoutMs,
    ): JsonObject = request("GET", path, null, token, timeoutMs)

    /** Convenience POST with a JSON body. */
    suspend fun post(
        path: String,
        body: JsonObject? = null,
        token: String? = null,
        timeoutMs: Int = this.timeoutMs,
    ): JsonObject = request("POST", path, body, token, timeoutMs)

    private fun readBody(
        connection: HttpURLConnection,
        status: Int,
    ): String {
        val stream = if (status in 200..299) {
            try {
                connection.inputStream
            } catch (e: IOException) {
                null
            }
        } else {
            connection.errorStream
        } ?: return ""
        return stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
    }

    private fun parseObject(
        raw: String,
        method: String,
        path: String,
    ): JsonObject {
        if (raw.isBlank()) return buildJsonObject { }
        val element = try {
            ControlJson.parseToJsonElement(raw)
        } catch (e: Exception) {
            throw ProtocolException(
                "server returned non-JSON response",
                "invalid_response",
            )
        }
        return try {
            element.jsonObject
        } catch (e: IllegalArgumentException) {
            throw ProtocolException(
                "server returned a non-object response",
                "invalid_response",
            )
        }
    }

    private fun mapHttpError(
        status: Int,
        raw: String,
        method: String,
        path: String,
    ): ProtocolException {
        var code = "http_$status"
        var message = "${method.uppercase()} $path failed"
        if (raw.isNotBlank()) {
            try {
                val body = ControlJson.parseToJsonElement(raw).jsonObject
                body["error_code"]?.jsonPrimitive?.contentOrNull?.let {
                    if (it.isNotBlank()) code = it
                }
                body["error"]?.jsonPrimitive?.contentOrNull?.let {
                    if (it.isNotBlank()) message = it.take(300)
                }
            } catch (e: Exception) {
                // Ignore: fall back to the generic message.
            }
        }
        message = redactTokenText(message)
        return when (status) {
            401 -> AuthenticationException(message, code)
            403 -> AuthorizationException(message, code)
            429 -> RateLimitException(message, code)
            else -> ProtocolException(message, code, status)
        }
    }
}
