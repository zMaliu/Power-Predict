# 工作流 / HTTP 入参出参对照（定版）

> **工作流的入参、出参已定，不要改。**  
> HTTP 节点只负责把工作流入参转给模型，再把两路结果交给代码节点，拼成工作流出参。

---

## 一、工作流入参（开始节点，已定）

| 变量 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `anchor_date` | 文本 string | 否 | 锚定日 `YYYY-MM-DD`；不传则模型用服务器当天 |
| `weather` | 对象/数组（段落） | **是** | 未来气象列表，天数可变（如 7 天），不必满 18 |

### `weather` 元素格式（后端传入）

```json
[
  {
    "date": "2026-09-14",
    "tem": 27.3,
    "tem_max": 31.0,
    "tem_min": 28.5,
    "rhum": 84.8,
    "precip": 2.5,
    "wind_speed": 1.8,
    "weather_type": "雷阵雨"
  }
]
```

---

## 二、工作流出参（输出节点，已定）

| 变量 | 类型 | 说明 |
|------|------|------|
| `prediction_json` | string | 合并后的 18 天预测 JSON 字符串 |
| `status` | string | `success` / `error` |
| `error_message` | string | 失败原因；成功可为空字符串 |

### `prediction_json` 建议内容（代码节点拼好再 stringify）

```json
[
  {
    "date": "2026-09-14",
    "全行业用电": 3176.08,
    "城乡居民用电": 929.14,
    "全社会用电": 4105.22
  }
]
```

共 18 条；`全社会用电 = 全行业用电 + 城乡居民用电`。

---

## 三、HTTP 节点（两个并行，入参相同）

工作流 **不要改入参**；两个 HTTP 的 Body 都直接用开始节点的变量。

| 项 | 全行业 | 城乡居民 |
|----|--------|----------|
| 方法 | POST | POST |
| URL | `http://<IP>:<端口1>/predict/industry` | `http://<IP>:<端口2>/predict/resident` |
| Header | `Content-Type: application/json` | 同左 |

### HTTP 请求 Body（两边一样）

```json
{
  "anchor_date": "{{anchor_date}}",
  "weather": {{weather}}
}
```

说明：

- 对应工作流入参，**不再增加字段**
- `weather` 天数不够 18 也可以，模型会补估并仍返回 18 天电量

---

## 四、HTTP 响应（两路形态）

### 成功时共同外壳

```json
{
  "ok": true,
  "target": "industry 或 resident",
  "target_name": "全行业用电 或 城乡居民用电",
  "anchor_date": "2026-09-13",
  "forecast_start": "2026-09-14",
  "forecast_end": "2026-10-01",
  "horizon": 18,
  "weather_days_received": 7,
  "weather_days_filled": 11,
  "forecast": [ ... ]
}
```

### 行业 `forecast[]` 单项

```json
{ "date": "2026-09-14", "全行业用电": 3176.08 }
```

### 居民 `forecast[]` 单项

```json
{ "date": "2026-09-14", "城乡居民用电": 929.14 }
```

### 失败时

HTTP 非 2xx，或 body 类似：

```json
{ "detail": "错误说明" }
```

---

## 五、数据流（不要改工作流入出参）

```
开始
  anchor_date, weather          ← 工作流入参（已定）
    │
    ├─ HTTP 行业  Body = {anchor_date, weather}
    │     → 返回 forecast（全行业用电 × 18）
    │
    └─ HTTP 居民  Body = {anchor_date, weather}
          → 返回 forecast（城乡居民用电 × 18）
              │
         代码执行
          - 若任一路失败 → status=error, error_message=..., prediction_json=""
          - 成功则按 date 合并，算全社会，stringify → prediction_json
          - status=success, error_message=""
              │
         输出
          prediction_json, status, error_message   ← 工作流出参（已定）
```

---

## 六、对照总表

| 层级 | 入参 | 出参 |
|------|------|------|
| **工作流** | `anchor_date`, `weather` | `prediction_json`, `status`, `error_message` |
| **HTTP 请求** | `anchor_date`, `weather`（与工作流一致） | — |
| **HTTP 响应** | — | `ok` + `forecast`（18 天，单目标字段） |
| **代码节点** | 两路 HTTP 响应 | 写成工作流三个出参 |

**原则：工作流入参出参不动；HTTP 入参跟工作流入参对齐；合并与加和只在代码节点做。**
