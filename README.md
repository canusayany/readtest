<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1RLGql1QnLZ_bXSxtaXCZi1O3vlmIvPAa

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## 部署到 GitHub Pages (Deploy to GitHub Pages)

本项目已配置 GitHub Actions，可自动构建并部署到 GitHub Pages。

### 设置步骤

1.  **配置 API Key (Secrets)**:
    为了保护你的 API Key 不被公开，请将其设置为 GitHub Secrets：
    *   进入 GitHub 仓库页面。
    *   点击顶部导航栏的 **Settings**。
    *   在左侧菜单栏找到 **Secrets and variables** > **Actions**。
    *   点击 **New repository secret**。
    *   **Name**: `GEMINI_API_KEY`
    *   **Value**: 粘贴你的 Gemini API Key。
    *   点击 **Add secret**。

2.  **推送代码**:
    *   将你的更改推送到 `main` 或 `master` 分支。
    *   点击仓库的 **Actions** 标签页，你可以看到 "Deploy to GitHub Pages" 的工作流正在运行。

3.  **启用 GitHub Pages**:
    *   等待 Action 运行成功（通常显示绿色对勾）。Action 会自动创建一个 `gh-pages` 分支。
    *   进入仓库 **Settings**。
    *   在左侧菜单找到 **Pages**。
    *   在 **Build and deployment** > **Source** 中选择 **Deploy from a branch**。
    *   在 **Branch** 选择 `gh-pages` 并确保文件夹是 `/(root)`。
    *   点击 **Save**。

4.  **访问应用**:
    *   页面顶部会显示 "Your site is live at..." 的链接。
    *   点击该链接即可访问你的应用并分享给朋友。
