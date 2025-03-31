package com.aicheck.sms

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.provider.Telephony
import android.util.Log
import java.util.regex.Pattern

class SmsReceiver : BroadcastReceiver() {
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
                }
            }
        }
    }
}