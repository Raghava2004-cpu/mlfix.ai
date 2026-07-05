import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let daemonProcess: ChildProcess | undefined;
let output: vscode.OutputChannel;
let statusBar: vscode.StatusBarItem;

interface FixResponse {
  episode_id: string;
  fixed_code: string;
  explanation: string;
  confidence: number;
  model_used: string;
  category: string;
  stages_run: string[];
  critic_approved: boolean;
  critic_issues: string[];
  executed: boolean;
  execution_success: boolean | null;
  execution_stderr: string | null;
  judge_success: boolean;
  judge_reason: string;
  total_tokens_in: number;
  total_tokens_out: number;
  examples_used: number;
}

function getConfig() {
  const cfg = vscode.workspace.getConfiguration('mlfix');
  return {
    port: cfg.get<number>('daemonPort', 8765),
    daemonPath: cfg.get<string>('daemonPath', ''),
  };
}

function daemonUrl(port: number): string {
  return `http://127.0.0.1:${port}`;
}

function resolveDaemonPath(context: vscode.ExtensionContext, configuredPath: string): string {
  if (configuredPath.trim()) {
    return configuredPath.trim();
  }

  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    const candidate = path.join(folder.uri.fsPath, 'daemon');
    if (fs.existsSync(path.join(candidate, 'pyproject.toml'))) {
      return candidate;
    }
  }

  return path.resolve(context.extensionPath, '..', 'daemon');
}

async function syncBackendCommand() {
  const { port } = getConfig();
  try {
    const r = await fetch(`${daemonUrl(port)}/sync`, { method: 'POST' });
    if (!r.ok) throw new Error(`daemon returned ${r.status}`);
    const data = await r.json() as {
      ok: boolean;
      priors_merged?: number;
      arms_received?: number;
      reason?: string;
    };
    if (data.ok) {
      vscode.window.showInformationMessage(
        `mlfix: synced ${data.arms_received ?? 0} arms from backend, merged ${data.priors_merged ?? 0} locally.`,
      );
    } else {
      vscode.window.showWarningMessage(`mlfix sync: ${data.reason ?? 'failed'}`);
    }
  } catch (e: any) {
    vscode.window.showErrorMessage(`mlfix sync failed: ${e.message}`);
  }
}

async function waitForDaemon(port: number, timeoutMs = 20000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(`${daemonUrl(port)}/health`);
      if (r.ok) {
        return true;
      }
    } catch {
      // not up yet
    }
    await new Promise(res => setTimeout(res, 300));
  }
  return false;
}

type HealthState = 'ok' | 'no-backend' | 'not-ready' | 'down';

async function pollHealth(port: number): Promise<HealthState> {
  try {
    const r = await fetch(`${daemonUrl(port)}/health`);
    if (!r.ok) return 'not-ready';
    const data = await r.json() as {
      pipeline_ready: boolean;
      backend_enabled: boolean;
    };
    if (!data.pipeline_ready) return 'not-ready';
    return data.backend_enabled ? 'ok' : 'no-backend';
  } catch {
    return 'down';
  }
}

function renderStatusBar(state: HealthState) {
  const map: Record<HealthState, { icon: string; tooltip: string; color?: vscode.ThemeColor }> = {
    'ok':          { icon: '$(check) mlfix',        tooltip: 'mlfix ready (cloud sync on). Click for options.' },
    'no-backend':  { icon: '$(cloud-offline) mlfix', tooltip: 'mlfix ready (local only, backend disabled). Click for options.' },
    'not-ready':   { icon: '$(warning) mlfix',      tooltip: 'mlfix daemon up but pipeline not ready. Check GEMINI_API_KEY.', color: new vscode.ThemeColor('statusBarItem.warningBackground') },
    'down':        { icon: '$(error) mlfix',        tooltip: 'mlfix daemon is not reachable. Click for options.', color: new vscode.ThemeColor('statusBarItem.errorBackground') },
  };
  const cfg = map[state];
  statusBar.text = cfg.icon;
  statusBar.tooltip = cfg.tooltip;
  statusBar.backgroundColor = cfg.color;
  statusBar.command = 'mlfix.showMenu';
  statusBar.show();
}

function startHealthPolling(port: number, context: vscode.ExtensionContext) {
  const tick = async () => renderStatusBar(await pollHealth(port));
  tick();
  const handle = setInterval(tick, 15_000); // every 15s
  context.subscriptions.push({ dispose: () => clearInterval(handle) });
}

function startDaemon(context: vscode.ExtensionContext) {
  const { port, daemonPath } = getConfig();
  const cwd = resolveDaemonPath(context, daemonPath);

  output.appendLine(`Starting daemon in: ${cwd}`);
  output.appendLine(`Expected port: ${port}`);

  if (!fs.existsSync(cwd)) {
    output.appendLine(`Daemon path does not exist: ${cwd}`);
    output.appendLine('Set mlfix.daemonPath to the daemon folder, for example D:\\mlfix\\daemon.');
    vscode.window.showWarningMessage('mlfix: daemon path not found. Set mlfix.daemonPath to D:\\mlfix\\daemon.');
    return;
  }

  daemonProcess = spawn('uv', ['run', 'mlfix-daemon'], {
    cwd,
    env: { ...process.env, MLFIX_PORT: String(port) },
    shell: process.platform === 'win32',
  });

  daemonProcess.stdout?.on('data', d => output.append(`[daemon] ${d}`));
  daemonProcess.stderr?.on('data', d => output.append(`[daemon] ${d}`));
  daemonProcess.on('exit', code => {
    output.appendLine(`[daemon] exited with code ${code}`);
    daemonProcess = undefined;
  });
  daemonProcess.on('error', err => {
    output.appendLine(`[daemon] failed to start: ${err.message}`);
    vscode.window.showErrorMessage(`mlfix: could not start daemon. Is 'uv' on PATH? (${err.message})`);
  });
}

function stopDaemon() {
  if (daemonProcess && !daemonProcess.killed) {
    output.appendLine('Stopping daemon...');
    daemonProcess.kill();
    daemonProcess = undefined;
  }
}

async function showDiffAndPrompt(
  original: string,
  fixed: string,
  explanation: string,
  confidence: number,
  model: string,
): Promise<'accept' | 'reject'> {
  const origDoc = await vscode.workspace.openTextDocument({ content: original, language: 'python' });
  const fixedDoc = await vscode.workspace.openTextDocument({ content: fixed, language: 'python' });

  await vscode.commands.executeCommand(
    'vscode.diff',
    origDoc.uri,
    fixedDoc.uri,
    `mlfix: Original <-> Fix (${model}, confidence ${(confidence * 100).toFixed(0)}%)`,
  );

  const choice = await vscode.window.showInformationMessage(
    `mlfix: ${explanation}`,
    { modal: false },
    'Accept', 'Reject',
  );
  return choice === 'Accept' ? 'accept' : 'reject';
}

async function runFixFlow(
  editor: vscode.TextEditor,
  range: vscode.Range,
  code: string,
  errorInput: string,
) {
  const { port } = getConfig();
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'mlfix: analyzing...' },
    async () => {
      try {
        const r = await fetch(`${daemonUrl(port)}/fix`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code,
            error: errorInput,
            language: editor.document.languageId,
          }),
        });
        if (!r.ok) {
          const errText = await r.text();
          throw new Error(`daemon returned ${r.status}: ${errText}`);
        }
        const data = await r.json() as FixResponse;
        output.appendLine('--- pipeline trace ---');
        output.appendLine(`category:    ${data.category}`);
        output.appendLine(`examples:    ${data.examples_used} retrieved from memory`);
        output.appendLine(`stages:      ${data.stages_run.join(' -> ')}`);
        output.appendLine(`model:       ${data.model_used}`);
        output.appendLine(`critic:      ${data.critic_approved ? 'approved' : 'rejected'}`);
        if (data.critic_issues.length) {
          output.appendLine(`  issues:    ${data.critic_issues.join('; ')}`);
        }
        if (data.executed) {
          output.appendLine(`execution:   ${data.execution_success ? 'passed' : 'failed'}`);
          if (data.execution_stderr) {
            output.appendLine(`  stderr:    ${data.execution_stderr.slice(0, 300)}`);
          }
        } else {
          output.appendLine(`execution:   skipped (not runnable in sandbox)`);
        }
        output.appendLine(`judge:       ${data.judge_success ? 'SUCCESS' : 'FAILED'} - ${data.judge_reason}`);
        output.appendLine(`tokens:      ${data.total_tokens_in} in / ${data.total_tokens_out} out`);
        
        if (!data.judge_success) {
          const proceed = await vscode.window.showWarningMessage(
            `mlfix: pipeline judged the fix as failed - ${data.judge_reason}. Show it anyway?`,
            'Show', 'Cancel',
          );
          if (proceed !== 'Show') {
            vscode.window.showInformationMessage('mlfix: fix discarded.');
            return;
          }
        }

        const decision = await showDiffAndPrompt(
          code, data.fixed_code, data.explanation, data.confidence,
          `${data.model_used} · ${data.category}`,
        );

        try {
          await fetch(`${daemonUrl(port)}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ episode_id: data.episode_id, decision }),
          });
          output.appendLine(`feedback sent: episode=${data.episode_id} decision=${decision}`);
        } catch (e: any) {
          output.appendLine(`WARNING: failed to send feedback: ${e.message}`);
        }

        if (decision === 'accept') {
          const edit = new vscode.WorkspaceEdit();
          edit.replace(editor.document.uri, range, data.fixed_code);
          const applied = await vscode.workspace.applyEdit(edit);
          if (!applied) {
            throw new Error('VS Code rejected the workspace edit');
          }
          await editor.document.save();
          vscode.window.showInformationMessage('mlfix: fix applied.');
        } else {
          vscode.window.showInformationMessage('mlfix: fix rejected. Thanks for the feedback.');
        }
      } catch (e: any) {
        vscode.window.showErrorMessage(`mlfix failed: ${e.message}`);
        output.appendLine(`ERROR: ${e.message}`);
      }
    },
  );
}

async function fixSelectionCommand() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('mlfix: open a file first.');
    return;
  }
  const selection = editor.selection;
  const useWholeFile = selection.isEmpty;
  const range = useWholeFile
    ? new vscode.Range(0, 0, editor.document.lineCount, 0)
    : selection;
  const code = editor.document.getText(useWholeFile ? undefined : selection);

  if (!code.trim()) {
    vscode.window.showWarningMessage('mlfix: file/selection is empty.');
    return;
  }

  const errorInput = await vscode.window.showInputBox({
    prompt: 'Paste the error message or stack trace',
    placeHolder: 'e.g. RuntimeError: shape mismatch...',
    ignoreFocusOut: true,
  });
  if (!errorInput) return;

  await runFixFlow(editor, range, code, errorInput);
}

async function fixFromTerminalCommand() {
  const term = vscode.window.activeTerminal;
  if (!term) {
    vscode.window.showWarningMessage('mlfix: open a terminal first, run your code, then select the error text.');
    return;
  }

  const prevClipboard = await vscode.env.clipboard.readText();
  await vscode.commands.executeCommand('workbench.action.terminal.copySelection');
  const errorText = await vscode.env.clipboard.readText();
  await vscode.env.clipboard.writeText(prevClipboard); // restore

  if (!errorText || errorText === prevClipboard) {
    vscode.window.showWarningMessage(
      'mlfix: no text selected in the terminal. Highlight the error message first.',
    );
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('mlfix: open the file that produced the error.');
    return;
  }

  const useWholeFile = editor.selection.isEmpty;
  const range = useWholeFile
    ? new vscode.Range(0, 0, editor.document.lineCount, 0)
    : editor.selection;
  const code = editor.document.getText(useWholeFile ? undefined : editor.selection);

  await runFixFlow(editor, range, code, errorText);
}

async function checkDaemonCommand() {
  const { port } = getConfig();
  try {
    const r = await fetch(`${daemonUrl(port)}/health`);
    const data = await r.json() as {
      status: string;
      version: string;
      pipeline_ready: boolean;
      backend_enabled: boolean;
    };
    const backend = data.backend_enabled ? 'enabled' : 'disabled';
    vscode.window.showInformationMessage(
      `mlfix daemon: ${data.status} v${data.version}, pipeline ${data.pipeline_ready ? 'ready' : 'NOT ready'}, backend ${backend}`,
    );
  } catch (e: any) {
    vscode.window.showErrorMessage(`mlfix daemon unreachable: ${e.message}`);
  }
}

async function showUsageCommand() {
  const { port } = getConfig();
  try {
    const r = await fetch(`${daemonUrl(port)}/usage`);
    if (!r.ok) throw new Error(`daemon returned ${r.status}`);
    const data = await r.json() as {
      day: string;
      per_model: Array<{
        model: string;
        requests: number;
        tokens_in: number;
        tokens_out: number;
        daily_limit: number | null;
      }>;
    };

    output.appendLine(`--- usage for ${data.day || 'today'} ---`);
    if (data.per_model.length === 0) {
      output.appendLine('(no requests yet today)');
    } else {
      for (const m of data.per_model) {
        const limit = m.daily_limit ? `/${m.daily_limit}` : '';
        output.appendLine(
          `${m.model}: ${m.requests}${limit} requests, ${m.tokens_in} in / ${m.tokens_out} out tokens`,
        );
      }
    }
    output.show(true);
  } catch (e: any) {
    vscode.window.showErrorMessage(`mlfix usage failed: ${e.message}`);
  }
}

async function showBanditCommand() {
  const { port } = getConfig();
  try {
    const r = await fetch(`${daemonUrl(port)}/bandit`);
    if (!r.ok) throw new Error(`daemon returned ${r.status}`);
    const data = await r.json() as {
      arms: Array<{
        category: string;
        model: string;
        alpha: number;
        beta: number;
        estimated_success_rate: number;
        total_pulls: number;
      }>;
    };
    output.appendLine(`--- bandit stats ---`);
    if (data.arms.length === 0) {
      output.appendLine('(no arms yet — run some fixes first)');
    } else {
      for (const a of data.arms) {
        output.appendLine(
          `[${a.category}] ${a.model}: success_rate≈${(a.estimated_success_rate * 100).toFixed(1)}% ` +
          `(pulls=${a.total_pulls}, α=${a.alpha}, β=${a.beta})`,
        );
      }
    }
    output.show(true);
  } catch (e: any) {
    vscode.window.showErrorMessage(`mlfix bandit failed: ${e.message}`);
  }
}

async function showMenuCommand() {
  const pick = await vscode.window.showQuickPick(
    [
      { label: '$(wand) Fix selected code',      cmd: 'mlfix.fixSelection' },
      { label: '$(pulse) Check daemon status',   cmd: 'mlfix.checkDaemon' },
      { label: '$(graph) Show today\'s usage',   cmd: 'mlfix.showUsage' },
      { label: '$(rocket) Show bandit stats',    cmd: 'mlfix.showBandit' },
      { label: '$(sync) Sync with backend',      cmd: 'mlfix.syncBackend' },
      { label: '$(output) Show mlfix logs',      cmd: 'mlfix.showLogs' },
    ],
    { placeHolder: 'mlfix' },
  );
  if (pick) vscode.commands.executeCommand(pick.cmd);
}

async function showLogsCommand() {
  output.show(true);
}

async function resetWelcomeCommand(context: vscode.ExtensionContext) {
  await context.globalState.update('mlfix.hasSeenWelcome', false);
  vscode.window.showInformationMessage('mlfix: welcome flag reset. Reload the window to see it.');
}

async function showWelcomeIfNeeded(context: vscode.ExtensionContext) {
  const hasSeenWelcome = context.globalState.get<boolean>('mlfix.hasSeenWelcome', false);
  if (hasSeenWelcome) {
    return;
  }

  const choice = await vscode.window.showInformationMessage(
    'mlfix is ready. Select Python code, paste an error with Ctrl+Alt+F, or select terminal error text and press Ctrl+Alt+T.',
    'Check Daemon',
    'Show Commands',
  );

  await context.globalState.update('mlfix.hasSeenWelcome', true);

  if (choice === 'Check Daemon') {
    await vscode.commands.executeCommand('mlfix.checkDaemon');
  } else if (choice === 'Show Commands') {
    await vscode.commands.executeCommand('mlfix.showMenu');
  }
}

export async function activate(context: vscode.ExtensionContext) {
  output = vscode.window.createOutputChannel('mlfix');
  context.subscriptions.push(output);

  output.appendLine('mlfix activating...');
  startDaemon(context);

  const { port } = getConfig();
  const up = await waitForDaemon(port);
  if (up) {
    output.appendLine('Daemon is healthy.');
  } else {
    output.appendLine('Daemon did not become healthy within timeout.');
    vscode.window.showWarningMessage('mlfix: daemon did not start. Check the mlfix output panel.');
  }

  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  context.subscriptions.push(statusBar);

  context.subscriptions.push(
    vscode.commands.registerCommand('mlfix.fixSelection', fixSelectionCommand),
    vscode.commands.registerCommand('mlfix.fixFromTerminal', fixFromTerminalCommand),
    vscode.commands.registerCommand('mlfix.checkDaemon', checkDaemonCommand),
    vscode.commands.registerCommand('mlfix.showUsage', showUsageCommand),
    vscode.commands.registerCommand('mlfix.showBandit', showBanditCommand),
    vscode.commands.registerCommand('mlfix.syncBackend', syncBackendCommand),
    vscode.commands.registerCommand('mlfix.showMenu', showMenuCommand),
    vscode.commands.registerCommand('mlfix.showLogs', showLogsCommand),
    vscode.commands.registerCommand('mlfix.resetWelcome', () => resetWelcomeCommand(context)),
  );

  startHealthPolling(port, context);
  await showWelcomeIfNeeded(context);
}

export function deactivate() {
  stopDaemon();
}
