# RAG 知识库问答系统

基于 FastAPI 的 RAG (Retrieval-Augmented Generation) 知识库问答系统。

## 项目结构

```
rag-kb-qa/
├── app/
│   ├── main.py       # FastAPI 应用入口
│   ├── schemas.py    # 数据模型定义
│   └── rag.py        # RAG 核心逻辑
├── data/             # 知识库数据目录
└── requirements.txt  # 项目依赖
```

## 安装

```bash
cd rag-kb-qa
pip install -r requirements.txt
```

## 运行

```bash
uvicorn app.main:app --reload
```

访问 http://localhost:8000/docs 查看 API 文档。
