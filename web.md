# 前端需求与接口文档

## 1. 文档说明


10.62.200.40
- 服务名称：`YOLOv8-Seg-DeepSORT`
- 默认启动地址：`http://10.62.200.40:8000`
- 数据格式：`application/json`
- 真实业务输入：`视频流`
- 当前后端输入形式：单帧图像 `base64` 字符串，字段名为 `image`
- 前端对接方式：从视频流中持续截取帧图，并将单帧图像转为 `base64` 后调用后端接口
- 跨域说明：后端已开启 `CORS`，当前允许任意来源访问

---

## 2. 前端建设目标

前端建议围绕以下 3 类核心能力建设：

1. 服务状态查看
2. 视频流目标检测与追踪结果展示
3. 视频流画面标注结果预览与导出

---

## 3. 前端页面需求

## 3.1 首页/工作台

建议提供一个统一工作台页面，包含以下区域：

- 服务状态区
  - 显示后端服务是否可用
  - 显示模型是否加载成功
  - 显示追踪器是否初始化成功
- 视频源接入区
  - 支持选择本地视频文件
  - 支持接入摄像头视频流
  - 支持 HTTP 等流地址输入
  - 支持显示当前视频源状态
- 检测结果区
  - 展示当前帧检测结果
  - 展示处理耗时
  - 展示类别名称列表
  - 展示追踪框数据
  - 展示轮廓数据
- 标注预览区
  - 展示当前帧或连续帧的标注画面
  - 支持查看原始视频帧与标注帧对比
  - 支持截图导出当前标注帧
- 调试信息区
  - 展示接口请求参数摘要
  - 展示接口响应原始 JSON
  - 展示错误信息

## 3.2 推荐交互流程

1. 页面初始化时调用 `GET /health` 检查服务状态。
2. 用户接入视频流后，前端开始读取视频帧。
3. 前端按设定频率对视频流抽帧，并将当前帧转为 base64。
4. 调用 `POST /track` 获取当前帧的结构化检测与追踪结果。
5. 如需直接展示后端绘制后的画面，则调用 `POST /track_multiple` 获取标注后的当前帧图像。
6. 前端持续刷新结果区域，形成“视频流检测”效果。

---

## 4. 前端功能需求

## 4.1 基础功能

- 页面启动后自动检查后端状态
- 支持视频流接入
- 支持从视频流持续抽帧
- 支持帧图转 base64 后提交
- 支持当前帧结果可视化展示
- 支持标注帧预览
- 支持错误提示

## 4.2 建议增强功能

- 接入前校验视频源可用性
- 支持设置抽帧频率或推理频率
- 请求处理中显示 loading
- 连续多次请求时防重复点击
- 支持复制返回 JSON
- 支持导出当前标注帧
- 支持展示单帧请求耗时
- 支持显示实时 FPS 或接口吞吐情况

## 4.3 异常处理要求

- 后端不可达时，提示“服务未启动或网络不可用”
- 视频帧 base64 非法时，提示“帧图数据无效”
- 后端返回 500 时，展示后端返回的 `detail`
- 用户未接入视频源时，禁止启动检测

---

## 4.4 关键需求说明

当前项目的业务输入不是“用户上传一张图片”，而是“用户接入一个持续的视频流”。

但由于当前后端 `serve.py` 的接口定义仍是：

- 每次请求只接收一个 `image`
- `image` 的内容是单帧图像的 `base64`

因此前端需要按“视频流驱动、单帧请求”的方式实现：

1. 采集视频流
2. 从视频流中按频率抽帧
3. 将单帧转为 `base64`
4. 把这一帧提交给后端
5. 将返回结果叠加到视频播放区域或结果区域中

这点是前端实现时最重要的理解前提。

---

## 5. 接口总览

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | `GET` | 获取服务基本信息 |
| `/health` | `GET` | 获取服务健康状态 |
| `/track` | `POST` | 返回视频流当前帧的检测与追踪结构化结果 |
| `/track_multiple` | `POST` | 返回视频流当前帧的标注图像 |

---

## 6. 接口详细说明

## 6.1 获取服务信息

- 路径：`/`
- 方法：`GET`
- 用途：前端可用于确认服务已启动以及可用接口列表

### 响应示例

```json
{
  "title": "YOLOv8-Seg-DeepSORT",
  "endpoints": [
    "GET /",
    "GET /health",
    "POST /track",
    "POST /track_multiple"
  ]
}
```

### 前端处理建议

- 该接口适合作为简单连通性测试
- 一般不作为主业务接口

---

## 6.2 获取健康状态

- 路径：`/health`
- 方法：`GET`
- 用途：检查模型和追踪器是否完成初始化

### 响应示例

```json
{
  "status": "healthy",
  "components": {
    "model_loaded": true,
    "tracker_initialized": true,
    "model_has_detect_method": true,
    "tracker_has_update_method": true
  },
  "timestamp": 1713945600.123
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | `string` | 整体状态，通常为 `healthy` 或 `unhealthy` |
| `components.model_loaded` | `boolean` | 模型是否已加载 |
| `components.tracker_initialized` | `boolean` | 追踪器是否已初始化 |
| `components.model_has_detect_method` | `boolean` | 模型是否具备 `detect` 方法 |
| `components.tracker_has_update_method` | `boolean` | 追踪器是否具备 `update` 方法 |
| `timestamp` | `number` | 服务器返回时间戳 |

### 前端处理建议

- `status !== "healthy"` 时，禁止进入检测流程
- 页面顶部可常驻显示服务状态标签

### 注意事项

- 代码中声明了 `SystemStatus` 模型，但实际返回内容为 `status + components + timestamp`
- 前端联调时应以实际 JSON 返回结构为准

---

## 6.3.1 视频流单帧检测与追踪，返回推理后的坐标数据

- 路径：`/track`
- 方法：`POST`
- 用途：上传视频流中的当前帧，返回检测框、类别、轮廓、目标数量和处理耗时

### 请求头

```http
Content-Type: application/json
```

### 请求体

```json
{
  "image": "base64编码后的图片字符串"
}
```

### 请求字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | `string` | 是 | 视频流当前帧的 base64 字符串，不含文件对象，仅传字符串内容 |

### 响应示例

```json
{
  "tracker_boxes": [
    [120.0, 80.0, 260.0, 220.0, 1.0],
    [300.0, 100.0, 420.0, 260.0, 2.0]
  ],
  "class_names": ["apple", "orange"],
  "contour_list": [
    [[120, 80], [130, 90], [150, 120]],
    [[300, 100], [315, 115], [330, 150]]
  ],
  "object_count": 2,
  "processing_time": 0.143
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `tracker_boxes` | `number[][]` | 追踪框列表，通常每项前 5 位可理解为 `x1, y1, x2, y2, track_id` |
| `class_names` | `string[]` | 识别出的类别名称列表 |
| `contour_list` | `any[]` | 每个目标对应的轮廓点集 |
| `object_count` | `number` | 当前图片中识别出的目标数量 |
| `processing_time` | `number` | 本次处理耗时，单位秒 |

### 前端展示建议

- 使用表格展示 `class_names`
- 使用卡片展示 `object_count` 和 `processing_time`
- 使用视频播放器上方的 canvas 或叠加层按 `tracker_boxes` 和 `contour_list` 做二次绘制
- `track_id` 建议单独高亮显示，便于追踪同一目标
- 前端应把该接口理解为“当前帧坐标结果接口”，而不是普通图片识别接口

### 数据对齐说明

- `tracker_boxes`、`class_names`、`contour_list` 理论上按索引一一对应
- 前端渲染时应增加边界保护，避免数组长度不一致导致报错

---

## 6.3.2 视频流单帧检测与追踪，返回坐标渲染后的图片，前端直接显示即可，不需要前端额外绘制

- 路径：`/track_multiple`
- 方法：`POST`
- 用途：上传视频流中的当前帧，返回已绘制边界框、类别名、追踪 ID、轮廓的标注图

### 请求头

```http
Content-Type: application/json
```

### 请求体

```json
{
  "image": "base64编码后的图片字符串"
}
```

### 响应示例

```json
{
  "annotated_image": "base64编码后的标注图片字符串"
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `annotated_image` | `string` | 当前帧标注完成后的图片 base64 字符串 |

### 前端展示建议

- 将 `annotated_image` 拼成 `data:image/jpeg;base64,${annotated_image}` 后直接展示
- 支持“原始帧 / 标注帧”切换
- 支持导出当前帧标注图
- 前端应把该接口理解为“当前帧渲染结果接口”

### 注意事项

- 虽然后端代码声明了 `response_model=MultipleImagesRequest`，但实际返回的是 `annotated_image`
- 前端联调时应以实际返回 JSON 为准

---

## 7. 错误响应说明

后端主要通过 `HTTPException` 返回错误，前端统一按以下方式处理：

### 常见错误格式

```json
{
  "detail": "错误描述"
}
```

### 典型场景

| HTTP状态码 | 场景 | 前端建议 |
|---|---|---|
| `400` | 当前视频帧无法解码，base64 格式错误 | 提示用户检查视频流或重新取帧 |
| `500` | 图像处理失败、推理异常、标注异常 | 展示 `detail`，并允许重试 |

---

## 8. 前端字段定义建议

可在前端定义如下数据结构。

```ts
export interface HealthResponse {
  status: string;
  components: {
    model_loaded: boolean;
    tracker_initialized: boolean;
    model_has_detect_method: boolean;
    tracker_has_update_method: boolean;
  };
  timestamp: number;
}

export interface ImageRequest {
  image: string;
}

export interface TrackResponse {
  tracker_boxes: number[][];
  class_names: string[];
  contour_list: any[];
  object_count: number;
  processing_time: number;
}

export interface TrackMultipleResponse {
  annotated_image: string;
}

export interface ApiError {
  detail: string;
}
```

---

## 9. 前端调用示例

## 9.1 检查服务状态

```ts
async function fetchHealth() {
  const res = await fetch("http://10.62.200.40:8000/health");
  if (!res.ok) throw new Error("健康检查失败");
  return res.json();
}
```

## 9.2 调用当前帧坐标检测接口

```ts
async function trackFrame(base64Image: string) {
  const res = await fetch("http://10.62.200.40:8000/track", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      image: base64Image
    })
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "检测失败");
  return data;
}
```

## 9.3 调用当前帧标注图接口

```ts
async function fetchAnnotatedFrame(base64Image: string) {
  const res = await fetch("http://10.62.200.40:8000/track_multiple", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      image: base64Image
    })
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "获取标注图失败");
  return data;
}
```

---

## 10. 图片处理要求

前端接入视频流时建议做如下处理：

1. 接入本地视频、摄像头或网络视频流
2. 在前端播放器或隐藏 canvas 中持续抽取当前帧
3. 将当前帧转为 base64
4. 去掉前缀中的 `data:image/png;base64,` 或 `data:image/jpeg;base64,`
5. 仅将纯 base64 内容传给后端的 `image` 字段

也就是说，这里虽然字段名叫 `image`，但业务含义实际是“视频流中的一帧”。

示例：

```ts
function frameCanvasToBase64(canvas: HTMLCanvasElement): string {
  const result = canvas.toDataURL("image/jpeg");
  return result.includes(",") ? result.split(",")[1] : result;
}
```

---

## 11. 联调注意事项

- 当前服务默认地址为 `10.62.200.40:8000`
- `track_multiple` 和 `track` 接口处理逻辑接近，但返回结果不同，前者返回当前帧渲染后的图像，后者返回当前帧推理后的坐标和结构化信息
- `health` 和 `track_multiple` 的 `response_model` 与真实返回值不完全一致，前端请按实际 JSON 联调
- 若后端模型初始化较慢，前端应在页面加载时增加“服务启动中”提示
- 若视频帧分辨率较大，base64 体积会明显增大，前端应控制抽帧分辨率和发送频率
- 为避免请求堆积，前端应采用“上一帧请求完成后再发送下一帧”或“固定节流频率”的策略
- 如果前端需要连续实时显示，建议优先以 `track` 返回的数据在前端自行绘制；如果前端只需要快速看到标注结果，可直接使用 `track_multiple`

---

## 12. 建议前端页面结构

建议至少包含以下模块：

- 顶部状态栏
  - 服务状态
  - 模型状态
  - 追踪器状态
- 左侧视频接入面板
  - 视频源选择
  - 摄像头开启按钮
  - 流地址输入框
  - 开始检测按钮
  - 停止检测按钮
- 右侧结果面板
  - 当前视频画面
  - 当前帧目标数量
  - 当前帧处理耗时
  - 类别列表
  - 追踪框数据
  - 标注帧预览
- 底部调试面板
  - 原始响应 JSON
  - 错误信息

---

## 13. 结论

当前后端面向前端主要提供两类能力：

- `POST /track`：获取视频流当前帧的结构化检测与追踪结果
- `POST /track_multiple`：获取视频流当前帧的标注图像结果

前端实现时，建议优先完成以下最小可用版本：

1. 健康检查
2. 视频源接入
3. 视频流抽帧
4. 当前帧检测结果展示
5. 当前帧标注图展示
6. 错误提示

这套能力即可支撑一个完整的视频流检测与追踪可视化页面。
