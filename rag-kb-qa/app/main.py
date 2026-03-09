from fastapi import FastAPI
from app.api.routes import router

# 创建FastAPI应用
app = FastAPI(title="RAG KB QA MVP")

# 注册路由
app.include_router(router)



