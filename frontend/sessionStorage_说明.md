# SessionStorage 功能实现说明

## 功能概述

根据图片中的要求，我们实现了 `sessionStorage` 功能，让用户在 `chat.html`（文件 A）和 `text_watermark.html`（文件 B）之间导航时保持页面状态，但在新会话或浏览器重启后清除状态。

## 实现特性

### 1. 会话级存储

- 数据只在当前浏览器会话中有效
- 浏览器窗口关闭时自动清除
- 新会话开始时自动清除所有状态

### 2. 页面导航保持

- 在同一浏览器窗口中，从页面 A 导航到页面 B 再返回页面 A 时，数据不会丢失
- 支持实时保存和恢复页面状态

### 3. 浏览器重启清除

- 浏览器重启后，下次访问页面时所有状态都会被清除
- 自动检测新会话并清除历史数据

## 实现细节

### chat.html 保存的状态

```javascript
const STORAGE_KEYS = {
  SELECTED_IMAGE: "chat_selected_image", // 选中的图片
  CHAT_HISTORY: "chat_history", // 聊天历史记录
  MODEL_SELECT: "chat_model_select", // 模型选择
  PROCESSING_STATUS: "chat_processing_status", // 处理状态
  WATERMARK_PROGRESS: "chat_watermark_progress", // 水印进度
  BINARY_IDENTITY: "chat_binary_identity", // 二进制序列
  HIGHLIGHT_TEXT: "chat_highlight_text", // 高亮文本
};
```

### text_watermark.html 保存的状态

```javascript
const STORAGE_KEYS = {
  INPUT_TEXT: "text_watermark_input_text", // 输入文本
  IDENTITY_BINARY: "text_watermark_identity_binary", // 身份二进制序列
  INITIAL_CAPTION: "text_watermark_initial_caption", // 初始描述
  COLORED_TEXT: "text_watermark_colored_text", // 带颜色的文本
  EXTRACTED_BINARY: "text_watermark_extracted_binary", // 提取的二进制序列
  METRICS: "text_watermark_metrics", // 验证指标
  STATUS_MESSAGE: "text_watermark_status_message", // 状态消息
};
```

## 保存时机

### 自动保存

1. **页面卸载时** (`beforeunload` 事件)
2. **页面隐藏时** (`visibilitychange` 事件)
3. **用户输入时** (防抖处理，1 秒后保存)
4. **模型选择改变时**
5. **验证完成后** (保存指标数据)

### 手动保存

- 导航到其他页面时自动保存当前状态

## 恢复时机

### 自动恢复

1. **页面加载完成时** (`DOMContentLoaded` 事件)
2. 检查是否有保存的状态数据
3. 如果有，自动恢复到页面元素中

## 新会话检测

```javascript
function isNewSession() {
  return (
    !sessionStorage.getItem(STORAGE_KEYS.CHAT_HISTORY) &&
    !sessionStorage.getItem(STORAGE_KEYS.SELECTED_IMAGE)
  );
}

// 页面加载时检查是否是新会话
if (isNewSession()) {
  console.log("检测到新会话，清除所有状态");
  clearPageState();
}
```

## 使用方法

### 1. 正常使用流程

1. 在 `chat.html` 中选择图片、输入消息、选择模型
2. 导航到 `text_watermark.html` 进行水印验证
3. 返回 `chat.html` 时，所有状态都会自动恢复
4. 关闭浏览器后重新打开，所有状态都会被清除

### 2. 测试功能

访问 `session_test.html` 页面可以测试 sessionStorage 功能：

- 输入内容后导航到其他页面
- 返回时查看内容是否恢复
- 关闭浏览器重新打开，检查内容是否被清除

## 技术实现

### 核心函数

#### 保存状态

```javascript
function savePageState() {
  try {
    // 保存各种页面状态到 sessionStorage
    sessionStorage.setItem(key, value);
    console.log("页面状态已保存到 sessionStorage");
  } catch (error) {
    console.error("保存页面状态失败:", error);
  }
}
```

#### 恢复状态

```javascript
function restorePageState() {
  try {
    // 从 sessionStorage 恢复各种页面状态
    const savedValue = sessionStorage.getItem(key);
    if (savedValue) {
      // 恢复到页面元素
    }
    console.log("页面状态恢复完成");
  } catch (error) {
    console.error("恢复页面状态失败:", error);
  }
}
```

#### 清除状态

```javascript
function clearPageState() {
  try {
    Object.values(STORAGE_KEYS).forEach((key) => {
      sessionStorage.removeItem(key);
    });
    console.log("页面状态已清除");
  } catch (error) {
    console.error("清除页面状态失败:", error);
  }
}
```

## 浏览器兼容性

- 支持所有现代浏览器
- 使用标准的 Web Storage API
- 自动处理存储限制和错误情况

## 注意事项

1. **存储限制**: sessionStorage 有存储大小限制（通常 5-10MB）
2. **安全性**: 数据只存储在客户端，不会发送到服务器
3. **隐私**: 浏览器关闭后数据自动清除，保护用户隐私
4. **性能**: 使用防抖处理避免频繁保存影响性能

## 调试信息

所有操作都会在浏览器控制台输出调试信息：

- 保存状态成功/失败
- 恢复状态成功/失败
- 新会话检测
- 清除状态操作

可以通过浏览器开发者工具的 Console 面板查看这些信息。
