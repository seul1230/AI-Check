package com.aicheck.call

import android.content.Context
import android.os.FileObserver
import android.util.Log
import com.aicheck.DeepVoiceDetector
import com.aicheck.DeepVoiceDetectorWithChaquopy
import com.aicheck.WavConverter2
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.File
import java.io.IOException

class CallRecordingFileObserver(
    path: String,
    private val deepVoiceDetector: DeepVoiceDetectorWithChaquopy,
    private val context: Context,
    private val phoneNumber: String? // ✅ 전달 받기
) : FileObserver(path, CREATE or MODIFY or CLOSE_WRITE)
{
    private val TAG = "CallRecordingFileObserver"
    private val observedDirectory = File(path)
    private var lastSentTime = 0L
    private val INTERVAL_MS = 3 * 1000L // 3초

    override fun onEvent(event: Int, fileName: String?) {
        if (fileName == null) return

        val newFile = File(observedDirectory, fileName)

        when (event) {
            CREATE -> {
                Log.d(TAG, "녹음 시작한다: ${newFile.absolutePath}")
                Log.d(TAG, "새로운 녹음 파일 감지됨: ${newFile.absolutePath}")
                lastSentTime = System.currentTimeMillis()
            }

            MODIFY -> {
                Log.d(TAG, "녹음 변경됐다: ${newFile.absolutePath}")
                processNewData(newFile)
            }

            CLOSE_WRITE -> {
                Log.d(TAG, "녹음 완료됨: ${newFile.absolutePath}")
                saveWav(newFile)
            }
        }
    }

    private fun saveWav(m4aFile: File) {
        val wavFile = File(m4aFile.parent, "${getFileNameWithoutExtension(m4aFile)}.wav")

        val command = "-i \"${m4aFile.absolutePath}\" -ar 16000 -ac 1 -c:a pcm_s16le \"${wavFile.absolutePath}\""

         FFmpegKit.executeAsync(command) { session ->
             val returnCode = session.returnCode
             if (ReturnCode.isSuccess(returnCode)) {
                 Log.d("FFmpeg", "변환 성공: ${wavFile.absolutePath}")

                 // ✅ 딥보이스 모델 분석
                 try {
                     val result = deepVoiceDetector.detect(wavFile.absolutePath)
                     Log.d("DeepVoice", """
                    📣 딥보이스 탐지 결과
                    - 파일 이름: ${result["basename"]}
                    - 실제 라벨: ${result["true_label"]}
                    - 평균 세그먼트 확률: ${result["mean_segment_prob"]}
                    - 전체 딥페이크 확률: ${result["deepfake_prob_full"]}
                    - 딥페이크 여부: ${result["is_deepfake_full"]}
                """.trimIndent())
                     val isDeepfake = result["is_deepfake_full"] as? Boolean ?: false

                     if (isDeepfake) {
                         Log.d("DeepVoice", "🚨 딥페이크로 판단되어 서버에 전송합니다.")
                         sendPhishingResultToServer(context, result, phoneNumber ?: "알 수 없음")
                     } else {
                         Log.d("DeepVoice", "✅ 정상 음성으로 판단됨. 서버 전송 생략.")
                     }
                 } catch (e: Exception) {
                     Log.e("DeepVoice", "탐지 중 오류", e)
                 }

             } else {
                 Log.e("FFmpeg", "변환 실패: ${session.failStackTrace}")
             }
         }
     }

    private fun getFileNameWithoutExtension(file: File): String {
        val name = file.name
        val dotIndex = name.lastIndexOf('.')
        return if (dotIndex == -1) name else name.substring(0, dotIndex)
    }

    private fun processNewData(file: File) {
        if (!file.exists()) return

        val currentTime = System.currentTimeMillis()
        Log.d(TAG, "파일 감지됨: ${file.absolutePath}")

        if (currentTime - lastSentTime >= INTERVAL_MS) {
//            val outputWav = File(observedDirectory, "debug_audio.wav")
//            extractFromStartToNow(file, outputWav)
//            lastSentTime = currentTime
        } else {
            Log.d(TAG, "저장 조건 미충족. 데이터 처리 안 함")
        }
    }

    fun sendPhishingResultToServer(context: Context, result: Map<String, Any>, phoneNumber: String) {
        Log.d("PhishingUploader", "🚀 서버 전송 시작!")

        val accessToken = getAccessTokenFromPrefs(context)
        if (accessToken == null) {
            Log.e("PhishingUploader", "❌ accessToken 없음. 서버 전송 불가.")
            return
        }

        val score = (result["deepfake_prob_full"] as? Number)?.toFloat() ?: 0f

        Log.d("PhishingUploader", """
        📦 전송할 데이터:
        - phoneNumber: $phoneNumber
        - score: $score
    """.trimIndent())

        val json = JSONObject().apply {
            put("phoneNumber", phoneNumber)
            put("score", score.toDouble())
        }

        val jsonString = json.toString()
        Log.d("PhishingUploader", "📨 JSON 바디: $jsonString")

        val client = OkHttpClient()
        val mediaType = "application/json".toMediaType()
        val requestBody = jsonString.toRequestBody(mediaType)

        val request = Request.Builder()
            .url("https://j12a603.p.ssafy.io/aicheck/phishings")
            .addHeader("Authorization", "Bearer $accessToken")
            .post(requestBody)
            .build()

        Log.d("PhishingUploader", "🔐 요청 준비 완료. 서버로 전송 중...")

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("PhishingUploader", "❌ 서버 전송 실패: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string()
                if (response.isSuccessful) {
                    Log.d("PhishingUploader", "✅ 서버 전송 성공! 응답: $body")
                } else {
                    Log.e("PhishingUploader", "❌ 서버 응답 실패: ${response.code} / 응답 바디: $body")
                }
            }
        })
    }


    private fun getAccessTokenFromPrefs(context: Context): String? {
        val prefs = context.getSharedPreferences("TokenStorage", Context.MODE_PRIVATE)
        return prefs.getString("accessToken", null)
    }

}
