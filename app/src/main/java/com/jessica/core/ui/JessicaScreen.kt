package com.jessica.core.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.platform.LocalContext
import com.jessica.core.core.JessicaController
import com.jessica.core.navigation.JessicaNavigator
import com.jessica.core.navigation.JessicaPage
import com.jessica.core.modules.chat.ChatThreadStore
import com.jessica.core.modules.chat.ChatAttachmentStore
import com.jessica.core.modules.updater.UpdateState
import com.jessica.core.modules.updater.UpdateViewModel
import com.jessica.core.ui.chat.ChatRoute
import com.jessica.core.ui.chat.ChatViewModel
import com.jessica.core.ui.chat.JessicaDrawerContent
import com.jessica.core.ui.home.HomeScreen
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JessicaScreen() {
    val context = LocalContext.current
    val controller = remember { JessicaController(context.applicationContext) }
    val navigator = remember { JessicaNavigator() }
    val updateViewModel = remember { UpdateViewModel(context.applicationContext) }
    val chatViewModel = remember {
        ChatViewModel(
            threadStore = ChatThreadStore(context.applicationContext),
            attachmentStore = ChatAttachmentStore(context.applicationContext),
            messageExecutor = { request, history, attachments ->
                controller.executeChatMessage(request, history, attachments)
            }
        )
    }

    val chatState by chatViewModel.state.collectAsState()
    val updateState by updateViewModel.state.collectAsState()
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    var showDeleteDialog by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val page = navigator.currentPage.value
    val chatBusy = chatState.isRunning || chatState.isLoading

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

    BackHandler(enabled = page != JessicaPage.CHAT) {
        navigator.backHome()
    }

    fun navigate(pageToOpen: JessicaPage) {
        navigator.navigateTo(pageToOpen)
        scope.launch { drawerState.close() }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            JessicaDrawerContent(
                threads = chatState.threads,
                activeThreadId = chatState.activeThreadId,
                searchQuery = chatState.searchQuery,
                searchResults = chatState.searchResults,
                diagnosticError = chatState.debugSyncError,
                onSearch = chatViewModel::updateSearch,
                onDiagnosticSync = chatViewModel::refreshChatDiagnostics,
                busy = chatBusy,
                onNewChat = {
                    if (!chatBusy) {
                        chatViewModel.newChat()
                        navigate(JessicaPage.CHAT)
                    }
                },
                onThread = { id ->
                    if (!chatBusy) {
                        chatViewModel.openChat(id)
                        navigate(JessicaPage.CHAT)
                    }
                },
                onPreviewDeleted = { id ->
                    if (!chatBusy) {
                        chatViewModel.previewDeletedChat(id)
                        navigate(JessicaPage.CHAT)
                    }
                },
                onRestore = { id ->
                    if (!chatBusy) {
                        chatViewModel.restoreChat(id)
                        navigate(JessicaPage.CHAT)
                    }
                },
                onTasks = { navigate(JessicaPage.TASKS) },
                onReports = { navigate(JessicaPage.REPORTS) },
                onSettings = { navigate(JessicaPage.HOME) }
            )
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Text(when (page) {
                            JessicaPage.CHAT -> if (chatState.previewDeletedThreadId != null)
                                "Удалённый чат · просмотр" else "Jessica"
                            JessicaPage.TASKS -> "Задачи"
                            JessicaPage.REPORTS -> "Отчёты"
                            JessicaPage.HOME -> "Jessica Core"
                        })
                    },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Text("☰")
                        }
                    },
                    actions = {
                        if (page == JessicaPage.CHAT) {
                            IconButton(
                                onClick = { chatViewModel.newChat() },
                                enabled = !chatBusy
                            ) {
                                Text("＋")
                            }
                            if (chatState.previewDeletedThreadId != null || chatState.undoThreadId != null) {
                                IconButton(
                                    onClick = chatViewModel::undoDelete,
                                    enabled = !chatBusy,
                                    modifier = Modifier.semantics {
                                        contentDescription = "Отменить удаление чата"
                                    }
                                ) {
                                    Text("↶", style = androidx.compose.material3.MaterialTheme.typography.headlineSmall)
                                }
                            } else {
                                IconButton(
                                    onClick = { showDeleteDialog = true },
                                    enabled = !chatBusy && chatState.activeThreadId != null && chatState.previewDeletedThreadId == null,
                                    modifier = Modifier.semantics {
                                        contentDescription = "Удалить текущий чат"
                                    }
                                ) {
                                    Text("🗑")
                                }
                            }
                        }
                    }
                )
            }
        ) { padding ->
            Box(modifier = Modifier.padding(padding).fillMaxSize()) {
                when (page) {
                    JessicaPage.CHAT -> ChatRoute(viewModel = chatViewModel)
                    JessicaPage.HOME -> HomeScreen(
                        message = controller.message.value,
                        updateState = updateState,
                        onUpdate = updateViewModel::checkUpdate
                    )
                    JessicaPage.REPORTS -> ReportScreen(
                        reportStorage = controller.reportStorage,
                        onBack = navigator::backHome
                    )
                    JessicaPage.TASKS -> TaskScreen(
                        onBack = navigator::backHome
                    )
                }
            }
        }
    }
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Удалить чат?") },
            text = {
                Text(
                    "«" + (chatState.threads.firstOrNull { it.id == chatState.activeThreadId }?.title
                        ?: "Выбранный чат") + "» исчезнет из списка. " +
                    "В течение 60 минут его можно найти через поиск и восстановить стрелкой ↶. " +
                    "После этого сообщения и вложения будут удалены окончательно."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showDeleteDialog = false
                    chatViewModel.deleteActiveChat()
                }) { Text("Удалить") }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) { Text("Отмена") }
            }
        )
    }
}
