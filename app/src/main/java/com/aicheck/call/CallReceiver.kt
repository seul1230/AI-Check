package com.aicheck.call

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.database.Cursor
import android.net.Uri
import android.provider.CallLog
import android.telephony.TelephonyManager
import android.util.Log
import com.aicheck.App
import com.aicheck.DeepVoiceDetector
import com.aicheck.DeepVoiceDetectorWithChaquopy
import com.aicheck.VoicePhishingDetector

class CallReceiver(
    private val deepVoiceDetector: DeepVoiceDetectorWithChaquopy,  // 💡 추가
    private val voicePhishingDetector: VoicePhishingDetector
) : BroadcastReceiver() {
    companion object {
        private const val TAG = "CallReceiver"
        private var lastState: String = "" // 🔥 static 변수 → companion object
        private const val RECORDING_PATH = "/storage/emulated/0/Recordings/Call/" // 삼성폰 기준
        var lastPhoneNumber: String? = null
    }

    private var fileObserver: CallRecordingFileObserver? = null

    private fun registerCallRecordingObserver(context: Context) {
        if (fileObserver == null) {
            Log.d(TAG, "📡 통화 녹음 감지 시작 (FileObserver)!")

            fileObserver = CallRecordingFileObserver(
                RECORDING_PATH,
                deepVoiceDetector,
                context,
                lastPhoneNumber ?: "알 수 없음",
                voicePhishingDetector // ✅ 생성자에서 받은 거 사용
            )
            fileObserver?.startWatching()
        }
    }

    private fun unregisterCallRecordingObserver() {
        fileObserver?.stopWatching()
        fileObserver = null
        Log.d(TAG, "📡 통화 녹음 감지 중지 (FileObserver)!")
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (TelephonyManager.ACTION_PHONE_STATE_CHANGED == intent.action) {
            val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE) ?: return

            if (state == lastState) return
            lastState = state

            Log.d(TAG, "📞 전화 상태 변경 감지됨: $state")

            when (state) {
                TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                    Log.d(TAG, "📲 통화 중!")
                    registerCallRecordingObserver(context)
                }
                TelephonyManager.EXTRA_STATE_IDLE -> {
                    Log.d(TAG, "❌ 통화 종료됨!")
                    unregisterCallRecordingObserver()
                }
                TelephonyManager.EXTRA_STATE_RINGING -> {
                    handleRingingCall(context)
                }
            }
        }
    }

    private fun handleRingingCall(context: Context) {
        val phoneNumber = getLastIncomingNumber(context)
        Log.d(TAG, "☎️ 전화가 오고 있음! 번호: $phoneNumber")
    }

    private fun getLastIncomingNumber(context: Context): String {
        val callUri: Uri = CallLog.Calls.CONTENT_URI
        val cursor: Cursor? = context.contentResolver.query(
            callUri,
            null,
            null,
            null,
            CallLog.Calls.DATE + " DESC"
        )

        cursor?.use {
            if (it.moveToFirst()) {
                val numberIndex = it.getColumnIndex(CallLog.Calls.NUMBER)
                val number = it.getString(numberIndex)
                lastPhoneNumber = number // ✅ 저장
                return number
            }
        }
        return "알 수 없음"
    }

}