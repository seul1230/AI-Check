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
import androidx.lifecycle.lifecycleScope
import com.aicheck.biometric.BiometricCallback
import com.aicheck.call.CallReceiver
import com.aicheck.permission.PermissionManager
import com.aicheck.ui.WebViewManager
import com.aicheck.biometric.BiometricHelper
import com.aicheck.call.CallRecordingFileObserver
import com.aicheck.fcm.FCMTokenManager
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.coroutines.resume

class MainActivity : FragmentActivity() {
    private var callReceiver: CallReceiver? = null
    var webView: WebView? = null
    lateinit var biometricHelper: BiometricHelper
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var permissionManager: PermissionManager
    private lateinit var deepVoiceDetector: DeepVoiceDetectorWithChaquopy
    private lateinit var voicePhishingDetector: VoicePhishingDetector
    private lateinit var callObserver: CallRecordingFileObserver
    private val REQUEST_NOTIFICATION_PERMISSION = 100
    private val SMS_PERMISSION_REQUEST_CODE = 2001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        UrlModelManager.initialize(this)
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
        window.decorView.post {
            requestAllPermissionsSequentially()
        }
        voicePhishingDetector = (applicationContext as App).phishingDetector
        registerCallReceiver()
    }

    fun FragmentActivity.requestAllPermissionsSequentially() {
        lifecycleScope.launch {
            try {
                requestPhonePermissionSuspend()
                Log.d("권한", "1단계 완료 → 알림 권한 요청")

                requestNotificationPermissionSuspend()
                Log.d("권한", "2단계 완료 → SMS 권한 요청")

                requestSmsPermissionSuspend()
                Log.d("권한", "3단계 완료 → 저장소 권한 요청")

                requestStoragePermissionSuspend()
                Log.d("권한", "4단계 완료 → 모든 권한 허용 완료")
            } catch (e: Exception) {
                Log.e("권한", "🚫 권한 요청 중단됨: ${e.message}")
            }
        }
    }

    suspend fun FragmentActivity.requestPhonePermissionSuspend() {
        suspendCancellableCoroutine { cont ->
            permissionManager.requestPermissions(
                onGranted = { cont.resume(Unit) },
                onDenied = { cont.cancel(Exception("전화 권한 거부됨")) }
            )
        }
    }

    suspend fun FragmentActivity.requestNotificationPermissionSuspend() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val permission = Manifest.permission.POST_NOTIFICATIONS
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                suspendCancellableCoroutine<Unit> { cont ->
                    ActivityCompat.requestPermissions(
                        this,
                        arrayOf(permission),
                        REQUEST_NOTIFICATION_PERMISSION
                    )
                    notificationContinuation = cont
                }
            } else {
                requestFCMToken()
            }
        } else {
            requestFCMToken()
        }
    }


    suspend fun FragmentActivity.requestSmsPermissionSuspend() {
        val receiveGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS) == PackageManager.PERMISSION_GRANTED
        val readGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS) == PackageManager.PERMISSION_GRANTED

        if (receiveGranted && readGranted) return

        suspendCancellableCoroutine<Unit> { cont ->
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECEIVE_SMS, Manifest.permission.READ_SMS),
                SMS_PERMISSION_REQUEST_CODE
            )
            smsContinuation = cont
        }
    }

    private suspend fun FragmentActivity.requestStoragePermissionSuspend() {
        suspendCancellableCoroutine<Unit> { cont ->
            permissionManager.requestStoragePermission {
                cont.resume(Unit)
            }
        }
    }

    private var notificationContinuation: CancellableContinuation<Unit>? = null
    private var smsContinuation: CancellableContinuation<Unit>? = null

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            REQUEST_NOTIFICATION_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    requestFCMToken()
                    notificationContinuation?.resume(Unit)
                } else {
                    notificationContinuation?.cancel(Exception("알림 권한 거부됨"))
                }
            }
            SMS_PERMISSION_REQUEST_CODE -> {
                val granted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
                if (granted) {
                    smsContinuation?.resume(Unit)
                } else {
                    smsContinuation?.cancel(Exception("SMS 권한 거부됨"))
                }
            }
        }
    }


    private fun registerCallReceiver() {
        try {
            val interpreter = Interpreter(loadModelFile("models/deepvoice_model.tflite")) // 📌 파일명 정확히!
            deepVoiceDetector = DeepVoiceDetectorWithChaquopy(this, interpreter)

            // ✅ 3. CallReceiver 초기화 (여기서 넘겨야 함)
            callReceiver = CallReceiver(deepVoiceDetector, voicePhishingDetector)

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

    private var alreadyRequestedNotificationPermission = false

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
