## 🚨 常见误操作与补救方法

### 1️⃣ 不小心删掉了文件或代码怎么办？

```bash
git restore 路径/文件名
```

还原误删或误改的文件到最近一次提交的状态。

---

### 2️⃣ 提交错了怎么办？想撤销最近一次 commit？

✅ 保留代码改动，只撤销提交：

```bash
git reset --soft HEAD~1
```

✅ 不保留改动，彻底还原到上一次提交：

```bash
git reset --hard HEAD~1
```

---

### 3️⃣ 不小心 push 了不该提交的内容？

如果只是普通 push，可以重新修改后再 push：

```bash
# 修改代码 → add → commit → push 覆盖
```

如果你用了 `--force` 强推导致回滚问题：

```bash
git reflog       # 查看历史操作记录
git checkout 提交ID
```

然后新建分支保存：

```bash
git checkout -b backup-from-reflog
```

---

### 4️⃣ 如何回退到之前的版本？

```bash
git log --oneline      # 查看历史提交
git checkout 提交ID    # 临时回到那个版本
```

---

### 5️⃣ 拉取（pull）后出现冲突了怎么办？

Git 会提示你手动合并冲突的文件，编辑完后执行：

```bash
git add 冲突文件
git commit -m "解决冲突"
git push origin 分支名
```

---

### 6️⃣ 误删 `.gitignore`，想恢复？

```bash
git checkout HEAD -- .gitignore
```

---

## 🧭 实用命令速查

| 命令 | 说明 |
|------|------|
| `git status` | 查看当前修改和状态 |
| `git log --oneline` | 简洁查看历史提交 |
| `git reflog` | 查看所有历史操作（包含被回退的） |
| `git reset` | 回滚提交 |
| `git restore` | 恢复文件内容 |

---

## 🧠 推荐习惯

- 不确定时：先 `git status`
- 尽量分支开发，不直接改 main
- 每次 `push` 前先 `pull`
- 每次 commit 加上清晰说明

---
