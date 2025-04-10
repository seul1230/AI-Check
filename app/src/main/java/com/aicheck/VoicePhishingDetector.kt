package com.aicheck

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.flex.FlexDelegate

class VoicePhishingDetector(private val context: Context) {
    private lateinit var interpreter: Interpreter
    private lateinit var vocabulary: Map<String, Int>
    private val maxLength = 5962 // 모델의 최대 시퀀스 길이

    init {
        loadModel()
        loadVocabulary()
    }

    // TFLite 모델 로드
    private fun loadModel() {
        try {
            val modelFile = loadModelFile(context, "models/model2.tflite")

            val options = Interpreter.Options().apply {
                addDelegate(FlexDelegate()) // ✅ Flex 연산자용 delegate 추가!
            }

            interpreter = Interpreter(modelFile, options)
            Log.d("VoicePhishingDetector", "Model loaded successfully")
        } catch (e: Exception) {
            Log.e("VoicePhishingDetector", "Error loading model: ${e.message}")
        }
    }

    // assets에서 모델 파일 읽기
//    private fun loadModelFile(fileName: String): MappedByteBuffer {
//        val fileDescriptor = context.assets.openFd("model2.tflite")
//        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
//        val fileChannel = inputStream.channel
//        val startOffset = fileDescriptor.startOffset
//        val declaredLength = fileDescriptor.declaredLength
//        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
//    }

    private fun loadModelFile(context: Context, modelFileName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    private fun loadVocabulary() {
        try {
            Log.d("VoicePhishingDetector", "📥 vocab.json 로딩 시도 중")
            val inputStream = context.assets.open("helpers/vocab.json") // 여기서 파일 없으면 바로 예외
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            vocabulary = jsonObject.keys().asSequence().associateWith { jsonObject.getInt(it) }
            Log.d("VoicePhishingDetector", "✅ Vocabulary loaded with ${vocabulary.size} words")
        } catch (e: Exception) {
            Log.e("VoicePhishingDetector", "❌ vocab 로딩 실패: ${e.message}", e)
        }
    }

    // 텍스트를 시퀀스로 변환
    private fun textToSequence(text: String): FloatArray {
        // 공백으로 단어 분리
        val words = text.split(" ")
        val sequence = IntArray(maxLength) { 0 } // 패딩 포함 초기화

        // 단어를 인덱스로 변환 (vocab에 없는 단어는 0으로 처리)
        words.take(maxLength).forEachIndexed { index, word ->
            sequence[index] = vocabulary[word] ?: 0
        }

        // FloatArray로 변환 (모델 입력에 맞게)
        return sequence.map { it.toFloat() }.toFloatArray()
    }

    fun detectPhishing(jsonInput: String): Boolean {
        return try {
            val jsonObject = JSONObject(jsonInput)
            val text = jsonObject.getString("text")

            // 텍스트를 시퀀스로 변환
            val inputSequence = textToSequence(text)
            val input = arrayOf(inputSequence) // 모델 입력: [1, maxLength]

            // 모델 예측
            val output = Array(1) { FloatArray(2) } // 출력: [1, 2] (0: 정상, 1: 보이스피싱)
            interpreter.run(input, output)

            val normalScore = output[0][0]
            val phishingScore = output[0][1]
            val isPhishing = phishingScore > normalScore

            Log.d("VoicePhishingDetector", "분석 결과: 정상=$normalScore, 피싱=$phishingScore, isPhishing=$isPhishing")

            isPhishing
        } catch (e: Exception) {
            Log.e("VoicePhishingDetector", "Error detecting phishing: ${e.message}")
            false  // 오류 발생 시 기본적으로 정상으로 간주
        }
    }

    // 리소스 해제
    fun close() {
        interpreter.close()
    }
}

// 사용 예시 (Activity 또는 Fragment에서 호출)
fun exampleUsage(context: Context) {
    val detector = VoicePhishingDetector(context)
    val inputJson = """{'text': '나를 사랑하긴 해', 'quota': 15}"""
    val result = detector.detectPhishing(inputJson)

    detector.close()
}