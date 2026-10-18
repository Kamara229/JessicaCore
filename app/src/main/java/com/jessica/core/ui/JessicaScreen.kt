package com.jessica.core.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import com.jessica.core.core.JessicaController
import com.jessica.core.navigation.JessicaNavigator
import com.jessica.core.navigation.JessicaPage
import com.jessica.core.modules.chat.ChatHistoryStore
import com.jessica.core.modules.updater.UpdateState
import com.jessica.core.modules.updater.UpdateViewModel
import com.jessica.core.ui.chat.ChatRoute
import com.jessica.core.ui.chat.ChatViewModel
import com.jessica.core.ui.home.HomeScreen

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JessicaScreen() {
    val context = LocalContext.current
    val controller = remember { JessicaController(context.applicationContext) }
    val navigator = remember { JessicaNavigator() }
    val updateViewModel = remember { UpdateViewModel(context.applicationContext) }
    val chatViewModel = remember {
        ChatViewModel(
            historyStore = ChatHistoryStore(context.applicationContext),
            messageExecutor = { request, history ->
                controller.executeChatMessage(request, history)
            }
        )
    }

    val updateState by updateViewModel.state.collectAsState()
    val installationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) {
        updateViewModel.installationReturned()
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O ||
            context.packageManager.canRequestPackageInstalls()
        ) {
            updateViewModel.checkUpdate()
        } else {
            updateViewModel.permissionDenied()
        }
    }

    LaunchedEffect(updateState) {
        if (updateState is UpdateState.ReadyToInstall) {
            val intent = updateViewModel.prepareInstallation()
            if (intent != null) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                    !context.packageManager.canRequestPackageInstalls()
                ) {
                    updateViewModel.waitingPermission()
                    try {
                        permissionLauncher.launch(
                            Intent(
                                Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                                Uri.parse("package:${context.packageName}")
                            )
                        )
                    } catch (_: Exception) {
                        updateViewModel.permissionDenied()
                    }
                } else {
                    try {
                        updateViewModel.installing()
                        installationLauncher.launch(intent)
                    } catch (_: Exception) {
                        updateViewModel.installationReturned()
                    }
                }
            }
        }
    }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Jessica Core") }) }
    ) { padding ->
        Box(modifier = Modifier.padding(padding).fillMaxSize()) {
            when (navigator.currentPage.value) {
                JessicaPage.HOME -> HomeScreen(
                    message = controller.message.value,
                    updateState = updateState,
                    onChat = { navigator.navigateTo(JessicaPage.CHAT) },
                    onUpdate = { updateViewModel.checkUpdate() },
                    onReports = { navigator.navigateTo(JessicaPage.REPORTS) },
                    onTasks = { navigator.navigateTo(JessicaPage.TASKS) }
                )
                JessicaPage.CHAT -> ChatRoute(
                    viewModel = chatViewModel,
                    onBack = { navigator.backHome() }
                )
                JessicaPage.REPORTS -> ReportScreen(
                    reportStorage = controller.reportStorage,
                    onBack = { navigator.backHome() }
                )
                JessicaPage.TASKS -> TaskScreen(
                    onBack = { navigator.backHome() }
                )
            }
        }
    }
}
