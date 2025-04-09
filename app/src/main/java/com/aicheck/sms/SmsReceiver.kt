package com.aicheck.sms

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import com.aicheck.UrlModelManager
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.regex.Pattern

class SmsReceiver : BroadcastReceiver() {
    private val TAG = "SmsReceiver"
    companion object {
        private const val TAG = "SmsReceiver"

        // ✅ 정규식 escape 수정 완료
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
                    val url = matcher.group()
                    Log.d(TAG, "🌐 추출된 URL: $url")
                    try {
                        val result = UrlModelManager.detector.detect(url)
                        Log.d("검사결과", "결과: ${result.result}, 확률: ${result.probability}")
                        if (result.result == "불법 URL") {
                            sendBadUrlToServer(context, url, result.probability[1])
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "🚨 URL 탐지 중 오류", e)
                    }
                }
            }
        }
    }

    fun sendBadUrlToServer(context: Context, url: String, score: Float) {
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

    fun getAccessTokenFromPrefs(context: Context): String? {
        val prefs = context.getSharedPreferences("TokenStorage", Context.MODE_PRIVATE)
        return prefs.getString("accessToken", null)
    }

}