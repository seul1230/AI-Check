package com.example.myapplication

import android.os.Bundle
import android.util.Log
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.serialization.json.Json
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var voicePhishingDetector: VoicePhishingDetector
    private lateinit var deepVoiceDetector: DeepVoiceDetectorWithChaquopy
    private lateinit var interpreterUrlDetector: Interpreter  // ✅ URL 탐지용 TFLite 모델

    private lateinit var wordToIndex: Map<String, Int>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        try {
            val fileName = "data/mixed_siu"
            copyAssetToInternalStorage("$fileName.wav")

            // ✅ 모델 로드
            val interpreterDeepVoice = Interpreter(loadModelFile("models/deepvoice_model.tflite"))
            val interpreterVoicePhishing = Interpreter(loadModelFile("models/model2.tflite"))
            val interpreterSTT = Interpreter(loadModelFile("models/stt_float36.tflite"))
            interpreterUrlDetector = Interpreter(loadModelFile("models/mlp-256-128-64-v6-auc-2-1_epoch_50_lr_0.001_batch_32_none_final_2025-04-09_00-17-44.tflite"))  // ✅ URL 모델

            // ✅ vocab.json
            val vocabJson = assets.open("helpers/vocab.json").bufferedReader().use { it.readText() }
            wordToIndex = Json.decodeFromString(vocabJson)

            // ✅ 탐지기 구성
            voicePhishingDetector = VoicePhishingDetector(this, interpreterSTT, interpreterVoicePhishing, wordToIndex)
            deepVoiceDetector = DeepVoiceDetectorWithChaquopy(this, interpreterDeepVoice)

            // ✅ UI 구성
            val btnPredictDeepVoice = findViewById<Button>(R.id.btnPredictDeepVoice)
            val btnPredictVoicePhishing = findViewById<Button>(R.id.btnPredictVoicePhishing)
            val btnCheckUrl = findViewById<Button>(R.id.btnCheckUrl)
            val tvResult = findViewById<TextView>(R.id.tvResult)
            val etUrlInput = findViewById<EditText>(R.id.etUrlInput)
            val tvUrlResult = findViewById<TextView>(R.id.tvUrlResult)
            val etAudioFilename = findViewById<EditText>(R.id.etAudioFilename)

            btnPredictDeepVoice.setOnClickListener {
                val userInput = etAudioFilename.text.toString().trim()
                if (userInput.isEmpty()) {
                    tvResult.text = "⚠️ 파일명을 입력하세요 (예: real_sample1)"
                    return@setOnClickListener
                }
                try {
                    val audioFileName = "data/$userInput.wav"
                    copyAssetToInternalStorage(audioFileName)
                    val audioPath = "${filesDir.absolutePath}/$audioFileName"
                    val result = deepVoiceDetector.detect(audioPath)
                    val resultText = """
                        
                        
                        
                        
                        
                        
🎧 딥 보이스 탐지 결과
📁 파일 이름: ${result["basename"]}
🏷️ 실제 라벨: ${result["true_label"]}
📊 세그먼트 확률:
${result["segment_probs"]}

🌐 전체 딥페이크 확률: ${result["deepfake_prob_full"]}
✅ 전체 딥페이크 여부: ${result["is_deepfake_full"]}

📈 평균 세그먼트 확률: ${result["mean_segment_prob"]}
🧠 세그먼트/최종 딥페이크 여부: ${result["is_deepfake_segment"]}
                    """.trimIndent()
                    tvResult.text = resultText
                } catch (e: Exception) {
                    Log.e("MainActivity", "딥 보이스 탐지 오류", e)
                    tvResult.text = "❌ 오류 발생: ${e.message}"
                }
            }

            btnPredictVoicePhishing.setOnClickListener {
                try {
                    val audioPath = "${filesDir.absolutePath}/${fileName}.wav"
                    val (result, transcribedText) = voicePhishingDetector.detect(audioPath)
                    tvResult.text = "보이스피싱 예측 결과: $result\n텍스트: $transcribedText"
                } catch (e: Exception) {
                    Log.e("MainActivity", "보이스피싱 탐지 오류", e)
                    tvResult.text = "❌ 오류: ${e.message}"
                }
            }

            // ✅ URL 탐지 버튼
            btnCheckUrl.setOnClickListener {
                val url = etUrlInput.text.toString().trim()
                if (url.isEmpty()) {
                    tvUrlResult.text = "URL을 입력하세요."
                    return@setOnClickListener
                }

                try {
                    val python = Python.getInstance()
                    val scalerPath = copyAssetToInternalStorage("helpers/scaler_params_v6.json")
                    val pyModule = python.getModule("extractor_url")

                    val result = pyModule.callAttr("extract_and_scale", url, scalerPath)
                    val featureVector = result.asList().map { it.toJava(Float::class.java) }.toFloatArray()

                    val inputBuffer = ByteBuffer.allocateDirect(4 * featureVector.size)
                    inputBuffer.order(ByteOrder.nativeOrder())
                    featureVector.forEach { inputBuffer.putFloat(it) }
                    inputBuffer.rewind()

                    val outputBuffer = ByteBuffer.allocateDirect(4)
                    outputBuffer.order(ByteOrder.nativeOrder())

                    // ✅ 실제 URL 모델로 추론
                    interpreterUrlDetector.run(inputBuffer, outputBuffer)
                    outputBuffer.rewind()
                    val resultProb = outputBuffer.float

                    val label = if (resultProb >= 0.5f) "⚠️ 악성 URL 의심됨" else "✅ 정상 URL"
                    tvUrlResult.text = """   
📌 입력 URL: $url
📊 악성 확률: ${"%.4f".format(resultProb)}
$label
""".trimIndent()

                } catch (e: Exception) {
                    Log.e("UrlCheck", "🔍 URL 검사 중 오류", e)
                    tvUrlResult.text = "❌ 오류 발생: ${e.message}"
                }
            }

        } catch (e: Exception) {
            Log.e("MainActivity", "초기화 오류", e)
            findViewById<TextView>(R.id.tvResult).text = "초기화 오류: ${e.message}"
        }
    }

    private fun loadModelFile(modelFileName: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    private fun copyAssetToInternalStorage(fileName: String): String {
        val outFile = File(filesDir, fileName)

        // 🔑 필요한 디렉토리 생성
        outFile.parentFile?.mkdirs()

        if (!outFile.exists()) {
            assets.open(fileName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile.absolutePath
    }

}
