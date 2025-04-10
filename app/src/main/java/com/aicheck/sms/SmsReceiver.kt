package com.aicheck.sms

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import com.aicheck.UrlModelManager
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import org.json.JSONObject
import java.io.IOException
import java.util.regex.Pattern

class SmsReceiver : BroadcastReceiver() {
    companion object {
        private const val TAG = "SmsReceiver"

        private val urlPattern: Pattern = Pattern.compile(
            "(https?://)?(www\\.)?[a-zA-Z0-9\\-]+\\.[a-z]{2,}(/[\\w\\-._~:/?\\[\\]\\@!$&'()*+,;=%]*)?",
            Pattern.CASE_INSENSITIVE
        )
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (Telephony.Sms.Intents.SMS_RECEIVED_ACTION == intent.action) {
            val msgs = Telephony.Sms.Intents.getMessagesFromIntent(intent)
            for (msg in msgs) {
                val messageBody = msg.messageBody
                Log.d(TAG, "📩 수신된 메시지: $messageBody")

                val matcher = urlPattern.matcher(messageBody)
                while (matcher.find()) {
                    val rawUrl = matcher.group()
                    val fullUrl = if (rawUrl.startsWith("http")) rawUrl else "http://$rawUrl"
                    Log.d(TAG, "🌐 추출된 URL: $rawUrl")

                    try {
                        // 프로토콜 제거
                        val urlForModel = fullUrl.replaceFirst("^https?://".toRegex(), "")
                        Log.d(TAG, "🔍 모델 입력용 URL: $urlForModel")

                        val maliciousProb = UrlModelManager.detectUrl(context, urlForModel)
                        Log.d(TAG, "🤖 악성 확률: $maliciousProb")

                        if (maliciousProb >= 0.5f) {
                            sendBadUrlToServer(context, fullUrl, maliciousProb) // 서버에는 전체 URL 전송
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "🚨 URL 탐지 중 오류", e)
                    }
                }
            }
        }
    }

    private fun sendBadUrlToServer(context: Context, url: String, score: Float) {
        val accessToken = getAccessTokenFromPrefs(context)
        if (accessToken.isNullOrBlank()) {
            Log.e(TAG, "🚨 accessToken이 없습니다.")
            return
        }

        val client = OkHttpClient()

        val json = JSONObject().apply {
            put("url", url)
            put("score", score.toDouble())
        }

        val requestBody = RequestBody.create(
            "application/json; charset=utf-8".toMediaTypeOrNull(),
            json.toString()
        )

        val request = Request.Builder()
            .url("https://j12a603.p.ssafy.io/aicheck/urls")
            .addHeader("Authorization", "Bearer $accessToken")
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "❌ 서버 전송 실패: ${e.message}", e)
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    Log.d(TAG, "✅ 서버 전송 성공")
                } else {
                    Log.e(TAG, "⚠️ 서버 응답 오류: ${response.code}")
                }
            }
        })
    }

    private fun getAccessTokenFromPrefs(context: Context): String? {
        val prefs = context.getSharedPreferences("TokenStorage", Context.MODE_PRIVATE)
        return prefs.getString("accessToken", null)
    }
}