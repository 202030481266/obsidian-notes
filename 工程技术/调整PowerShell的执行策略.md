
默认的PowerShell不会运行大部分的没有签名的脚本。但是这在开发的时候就很鸡肋，比如激活一个虚拟环境的时候还要跑到cmd中执行激活，非常憨憨！

是 PowerShell 的执行策略（Execution Policy）导致的。默认情况下，为了安全起见，PowerShell 可能会限制执行本地未签名的脚本文件（`.ps1` 文件），而虚拟环境的激活脚本 `activate.ps1` 通常就是这种。

`cmd.exe` 使用的是 `.bat` 脚本（`.venv/Scripts/activate.bat`），它不受 PowerShell 执行策略的限制，所以可以在 `cmd` 中正常工作。

要在 PowerShell 中能够执行 `activate.ps1` 脚本（以及其他本地脚本），你需要修改 PowerShell 的执行策略。

**重要提示：** 降低执行策略会增加执行恶意脚本的风险。请确保你了解相关的安全风险。`RemoteSigned` 是一个相对平衡的选择，它允许执行本地脚本，但要求从网络下载的脚本必须经过签名。

以下是几种修改执行策略的方法，按推荐程度和权限要求排序：

**方法一：为当前用户设置执行策略（推荐，无需管理员权限）**

这种方法只影响当前登录的用户，通常不需要管理员权限，是比较推荐的方式。

1.  打开 PowerShell（不需要管理员身份）。
2.  执行以下命令来设置执行策略为 `RemoteSigned` 或 `Unrestricted`：
    *   **推荐 `RemoteSigned`：** 允许运行本地创建的脚本，要求下载的脚本具有可信发布者的签名。
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        ```
    *   **更宽松 `Unrestricted`：** 允许运行所有脚本，但会为从互联网下载的未签名脚本提示确认。（风险相对更高）
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
        ```
3.  系统可能会提示你确认更改，输入 `Y` 然后按 Enter。
4.  设置完成后，你应该可以在 PowerShell 中激活虚拟环境了。

**方法二：为当前 PowerShell 会话临时设置执行策略（最安全，临时性）**

这种方法只在当前的 PowerShell 窗口/会话中有效，关闭窗口后设置就会失效。它不需要管理员权限，也不会永久性地降低安全性。

1.  打开 PowerShell。
2.  执行以下命令：
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    # 或者使用 Unrestricted
    # Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
    ```
3.  现在你可以在这个 PowerShell 会话中激活虚拟环境。

**方法三：为本机所有用户设置执行策略（需要管理员权限）**

这种方法会影响这台计算机上的所有用户，**需要以管理员身份运行 PowerShell**。

1.  右键点击开始菜单，选择 "Windows PowerShell (管理员)" 或 "终端 (管理员)"。
2.  在管理员 PowerShell 窗口中，执行以下命令：
    *   **推荐 `RemoteSigned`：**
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
        ```
    *   **更宽松 `Unrestricted`：**
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope LocalMachine
        ```
    *   **完全绕过 `Bypass`：** 不阻止任何操作，也不发出警告或提示。（**风险最高，非常不推荐**）
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope LocalMachine
        ```
3.  系统可能会提示你确认更改，输入 `Y` 然后按 Enter。

**如何检查当前的执行策略？**

你可以使用以下命令查看不同作用域的执行策略设置：

```powershell
Get-ExecutionPolicy -List
```

**如何在 PowerShell 中激活虚拟环境？**

修改了执行策略后，你需要使用 PowerShell 的 `.ps1` 脚本来激活环境，并且通常需要使用 "点 सोर्स" (dot sourcing) 的方式来执行，这样它才能修改当前 Shell 的环境变量：

1.  导航到你的项目目录：
    ```powershell
    cd path\to\your\project
    ```
2.  执行激活脚本（注意脚本路径前的 `.` 和一个空格，以及使用 `.ps1` 后缀）：
    ```powershell
    . .\.venv\Scripts\activate.ps1
    ```
    *   或者，如果你的 venv 文件夹叫 `venv`（注意路径分隔符 `\`）：
    ```powershell
    . .\venv\Scripts\activate.ps1
    ```

执行成功后，你的 PowerShell 提示符前面应该会显示虚拟环境的名称（例如 `(.venv)`）。

**总结：**

1.  **推荐**使用 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` 来为当前用户启用本地脚本执行。
2.  修改策略后，使用 `. .\.venv\Scripts\activate.ps1` (注意前面的点和空格，以及 `.ps1` 后缀) 来激活虚拟环境。