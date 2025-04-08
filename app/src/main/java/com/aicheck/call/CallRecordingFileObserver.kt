package com.aicheck.call

import android.os.FileObserver
import android.util.Log
import com.aicheck.DeepVoiceDetector
import com.aicheck.DeepVoiceDetectorWithChaquopy
import com.aicheck.WavConverter2
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import java.io.File

class CallRecordingFileObserver(
    path: String,
    private val deepVoiceDetector: DeepVoiceDetectorWithChaquopy
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

    private fun extractFromStartToNow(inputFile: File, outputFile: File) {
        try {
            if (!inputFile.exists()) {
                Log.e(TAG, "파일이 존재하지 않음: ${inputFile.absolutePath}")
                return
            }

            Log.d(TAG, "FFmpegKit으로 오디오 실시간 변환 시작...")

            val inputPath = "\"${inputFile.absolutePath}\""
            val outputPath = "\"${outputFile.absolutePath}\""

            val ffmpegCommand = "-re -i $inputPath -movflags +faststart -use_editlist 0 -c copy -f wav $outputPath"

            FFmpegKit.executeAsync(ffmpegCommand) { session ->
                val returnCode = session.returnCode
                val output = session.output
                val error = session.failStackTrace

                Log.d(TAG, "FFmpeg 실행 결과: $output")
                Log.e(TAG, "FFmpeg 오류 로그: $error")

                if (ReturnCode.isSuccess(returnCode)) {
                    Log.d(TAG, "FFmpegKit 실시간 변환 완료: ${outputFile.absolutePath}")
                } else {
                    Log.e(TAG, "FFmpegKit 변환 실패: $returnCode")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "FFmpegKit 실행 중 오류 발생: ${e.message}")
        }
    }

    private fun sendToAI(audioData: ByteArray, originalFile: File) {
        Log.d(TAG, "🚀 AI 모델에 데이터 전송: ${audioData.size} 바이트")

        val wavFile = File(observedDirectory, "output.wav")
        convertM4AToWav(originalFile, wavFile)
        sendWavToAI(wavFile)
    }

    fun convertM4AToWav(inputFile: File, outputFile: File) {
        val command = "-i ${inputFile.absolutePath} -acodec pcm_s16le -ar 16000 -ac 1 -f wav ${outputFile.absolutePath}"
        Log.d(TAG, "FFmpegKit 변환 완료: ${outputFile.absolutePath}")
        // FFmpegKit.executeAsync(command) { ... }  // 원래 변환 수행하려면 여기에 넣어야 함
    }

    private fun sendWavToAI(wavFile: File) {
        Log.d(TAG, "AI 모델에 WAV 파일 전송: ${wavFile.absolutePath}")
        // TODO: 온디바이스 AI 연동 코드
    }

    private fun saveDebugWav(inputFile: File) {
        val debugWavFile = File(observedDirectory, "debug_audio.wav")
        convertM4AToWav(inputFile, debugWavFile)
        Log.d(TAG, "디버깅용 WAV 파일 저장 완료: ${debugWavFile.absolutePath}")
    }

    private fun saveDebugFile2(audioData: ByteArray) {
        try {
            val rawFile = File(observedDirectory, "debug_audio.raw")
            val wavFile = File(observedDirectory, "debug_audio_fixed.wav")
            WavConverter2.addWavHeader(rawFile, wavFile, 16000, 1, 16)
            Log.d(TAG, "WAV 파일 변환 완료: ${wavFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "WAV 변환 실패: ${e.message}")
        }
    }
}
