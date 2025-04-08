package com.aicheck

import android.Manifest
import android.content.Context
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.telephony.TelephonyManager
import android.util.Log
import android.webkit.WebView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import com.aicheck.biometric.BiometricCallback
import com.aicheck.call.CallReceiver
import com.aicheck.permission.PermissionManager
import com.aicheck.ui.WebViewManager
import com.aicheck.biometric.BiometricHelper
import com.aicheck.call.CallRecordingFileObserver
import com.aicheck.fcm.FCMTokenManager
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : FragmentActivity() {
    private var callReceiver: CallReceiver? = null
    var webView: WebView? = null
    lateinit var biometricHelper: BiometricHelper
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var permissionManager: PermissionManager
    private lateinit var deepVoiceDetector: DeepVoiceDetectorWithChaquopy
    private lateinit var callObserver: CallRecordingFileObserver
    private val REQUEST_NOTIFICATION_PERMISSION = 100
    private val SMS_PERMISSION_REQUEST_CODE = 2001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        sharedPreferences = getSharedPreferences("TokenStorage", Context.MODE_PRIVATE)

        biometricHelper = BiometricHelper(this, object : BiometricCallback {
            override fun onBiometricSuccess() {
                runOnUiThread {
                    webView?.evaluateJavascript("alert('생체 인증 성공!')", null)
                }
            }

            override fun onBiometricFailure(errorMessage: String) {
                runOnUiThread {
                    webView?.evaluateJavascript("alert('생체 인증 실패: $errorMessage')", null)
                }
            }
        })

        val webViewManager = WebViewManager(this)
        webView = webViewManager.setupWebView()
        setContentView(webView)
        Log.d("MainActivity", "onCreate 호출됨!")

        // ✅ PermissionManager 초기화
        permissionManager = PermissionManager(this)

        // ✅ 필수 권한 요청
        permissionManager.requestPermissions(
            onGranted = {
                Log.d("MainActivity", "✅ 모든 권한 허용됨!")
                registerCallReceiver()
            },
            onDenied = {
                Log.e("MainActivity", "❌ 필수 권한이 거부되었습니다.")
            }
        )
        permissionManager.requestStoragePermission()
        requestNotificationPermissionIfNeeded()
        requestSmsPermission()
    }

    private fun registerCallReceiver() {
        try {
            val interpreter = Interpreter(loadModelFile("deepvoice_model.tflite")) // 📌 파일명 정확히!
            deepVoiceDetector = DeepVoiceDetectorWithChaquopy(this, interpreter)

            // ✅ 3. CallReceiver 초기화 (여기서 넘겨야 함)
            callReceiver = CallReceiver(deepVoiceDetector)

            // ✅ 4. registerReceiver 호출
            val filter = IntentFilter(TelephonyManager.ACTION_PHONE_STATE_CHANGED)
            registerReceiver(callReceiver, filter)

            Log.d("MainActivity", "CallReceiver 등록 완료!")
        } catch (e: Exception) {
            Log.e("MainActivity", "CallReceiver 등록 실패", e)
        }
    }


    override fun onDestroy() {
        super.onDestroy()
        callReceiver?.let {
            unregisterReceiver(it)
            Log.d("MainActivity", "CallReceiver 해제됨!")
        }
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val permission = Manifest.permission.POST_NOTIFICATIONS
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(permission), REQUEST_NOTIFICATION_PERMISSION)
            } else {
                Log.d("FCM", "🔔 알림 권한 이미 허용됨 → FCM 토큰 발급 진행")
                requestFCMToken()
            }
        } else {
            Log.d("FCM", "🔔 Android 12 이하 → FCM 토큰 바로 발급")
            requestFCMToken()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQUEST_NOTIFICATION_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d("FCM", "✅ 알림 권한 허용됨 → FCM 토큰 발급")
                requestFCMToken()
            } else {
                Log.e("FCM", "❌ 알림 권한 거부됨 → FCM 토큰 발급 불가")
            }
        }
    }

    private fun requestFCMToken() {
        FCMTokenManager.getFCMToken(object : FCMTokenManager.TokenCallback {
            override fun onSuccess(token: String?) {
                val prefs = getSharedPreferences("TokenStorage", Context.MODE_PRIVATE)
                prefs.edit().putString("fcmToken", token).apply()
                Log.d("FCM", "🔑 저장된 FCM Token: $token")
            }

            override fun onFailure(e: Exception?) {
                Log.e("FCM", "FCM 토큰 발급 실패", e)
            }
        })
    }

    private fun requestSmsPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECEIVE_SMS, Manifest.permission.READ_SMS),
                SMS_PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun loadModelFile(modelFileName: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        mappedByteBuffer.order(ByteOrder.nativeOrder())
        inputStream.close()
        fileChannel.close()
        return mappedByteBuffer
    }

}
