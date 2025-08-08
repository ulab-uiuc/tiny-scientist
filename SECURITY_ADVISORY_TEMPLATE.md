# 🚨 GitHub Security Advisory 快速填写模板

## 📋 直接复制粘贴的内容

### Title (标题)
```
Path Traversal Vulnerability in PDF Review Function (CWE-22)
```

### Summary (摘要)
```
A path traversal vulnerability exists in the review_paper function of backend/app.py that allows attackers to read arbitrary files on the server by bypassing path validation.
```

### Description (详细描述)
```markdown
## Description
A critical path traversal vulnerability (CWE-22) has been identified in the `review_paper` function in `backend/app.py`. The vulnerability allows malicious users to access arbitrary PDF files on the server by providing crafted file paths that bypass the intended security restrictions.

## Impact
This vulnerability allows attackers to:
- Read any PDF file accessible to the server process
- Potentially access sensitive documents outside the intended directory
- Perform reconnaissance on the server's file system structure

## Vulnerable Code
The issue occurs in the `review_paper` function around line 744:

```python
if pdf_path.startswith("/api/files/"):
    # Safe path handling for API routes
    relative_path = pdf_path[len("/api/files/"):]
    generated_base = os.path.join(project_root, "generated")
    absolute_pdf_path = os.path.join(generated_base, relative_path)
else:
    absolute_pdf_path = pdf_path  # VULNERABLE: Direct use of user input
```

When `pdf_path` does not start with "/api/files/", the code directly uses the user-provided path without any validation, allowing path traversal attacks.

## Proof of Concept
An attacker can exploit this by sending a POST request to `/api/review` with a malicious `pdf_path`:

```bash
# Access system files
curl -X POST http://localhost:5000/api/review \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/etc/passwd"}'

# Access files via path traversal
curl -X POST http://localhost:5000/api/review \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "../../../sensitive/document.pdf"}'
```

## Affected Versions
All versions containing the vulnerable `backend/app.py` file.

## Solution
The vulnerability has been fixed by:
1. Restricting all paths to only allow `/api/files/` prefixes
2. Adding proper path validation using `os.path.abspath()`
3. Implementing boundary checks to ensure files remain within the allowed directory

## Fixed Code
```python
if pdf_path.startswith("/api/files/"):
    relative_path = pdf_path[len("/api/files/"):]
    generated_base = os.path.join(project_root, "generated")
    absolute_pdf_path = os.path.abspath(os.path.join(generated_base, relative_path))
    
    # Security check: ensure the file is within the allowed directory
    if not absolute_pdf_path.startswith(os.path.abspath(generated_base)):
        return jsonify({"error": "Access denied - path traversal not allowed"}), 403
else:
    # Reject all non-API paths to prevent arbitrary file access
    return jsonify({"error": "Invalid path - only /api/files/ paths are allowed"}), 403
```

## Credit
This vulnerability was discovered and reported by Ruizhe.
```

### Severity (严重性)
```
Severity: High
CVSS Score: 7.5
```

### CWE
```
CWE-22: Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
```

### Affected Product (受影响的产品)
```
Ecosystem: Other
Package name: tiny-scientist
Affected versions: < [fix version]
Patched versions: [fix version]
```

---

## 🚀 操作步骤

1. **访问**: https://github.com/ulab-uiuc/tiny-scientiest/security/advisories
2. **点击**: "New draft security advisory"
3. **复制粘贴**: 上面的内容到对应字段
4. **勾选**: "Request a CVE identifier for this advisory"
5. **点击**: "Create draft security advisory"
6. **审核**: 所有信息是否正确
7. **点击**: "Publish advisory"

## ⚠️ 重要提醒

- 发布前请仔细检查所有信息
- 确保修复代码已经提交
- 发布后无法撤销，请谨慎操作
- CVE 申请可能需要几天时间处理

## 📧 后续通知

发布后，GitHub 会自动：
- 通知所有关注者
- 发送安全通知给依赖项目
- 在项目页面显示安全警告
- 生成安全报告 