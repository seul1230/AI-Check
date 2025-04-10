package com.aicheck

import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException

object ClovaSpeechClient {
    private const val TAG = "ClovaSpeechClient"
    private const val API_KEY = "7a1ad20b32bf4ddbbd75ed8afb897e96"
    private const val URL = "https://clovaspeech-gw.ncloud.com/recog/v1/stt?lang=Kor"
    private val client = OkHttpClient()

    interface Callback {
        fun onSuccess(text: String)
        fun onFailure(error: String)
    }

    fun transcribe(audioFile: File, callback: Callback) {
        val requestBody = audioFile.inputStream().use {
            it.readBytes().toRequestBody("application/octet-stream".toMediaType())
        }

        val request = Request.Builder()
            .url(URL)
            .post(requestBody)
            .addHeader("Content-Type", "application/octet-stream")
            .addHeader("X-CLOVASPEECH-API-KEY", API_KEY)
            .build()

        client.newCall(request).enqueue(object : okhttp3.Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "요청 실패", e)
                callback.onFailure("요청 실패: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string()
                if (response.isSuccessful && body != null) {
                    try {
                        val json = JSONObject(body)
                        val text = json.optString("text", "")
                        callback.onSuccess(text)
                    } catch (e: Exception) {
                        Log.e(TAG, "JSON 파싱 오류", e)
                        callback.onFailure("응답 파싱 실패")
                    }
                } else {
                    Log.e(TAG, "API 호출 에러: ${response.code} - $body")
                    callback.onFailure("API 호출 실패: ${response.code}")
                }
            }
        })
    }
}