import os
import sys
import logging
import time
import uvicorn
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 添加模块路径
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
sys.path.append(str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="DanceVibe API",
    version="1.0.0",
    description="舞蹈姿态检测和评分系统",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 创建必要的目录
def create_directories():
    """创建必要的目录"""
    directories = ["uploads", "temp", "logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 创建目录: {directory}")


# 设置静态文件服务
def setup_static_files():
    """设置静态文件服务"""
    try:
        # 前端静态文件
        frontend_path = PROJECT_ROOT / "frontend"
        if frontend_path.exists():
            app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
            logger.info(f"✅ 静态文件目录: {frontend_path}")
        else:
            logger.warning(f"⚠️ 静态文件目录不存在: {frontend_path}")

        # 上传文件目录
        uploads_path = PROJECT_ROOT / "uploads"
        uploads_path.mkdir(exist_ok=True)
        app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

        logger.info("✅ 静态文件服务配置完成")

    except Exception as e:
        logger.error(f"❌ 静态文件服务配置失败: {e}")


# 注册路由
def register_routes():
    """注册API路由"""
    try:
        # 延迟导入以避免循环依赖
        from api.ws_handler import ws_router
        from api.pose_api import router as pose_router
        from api.score_api import router as score_router

        # WebSocket路由 - 添加prefix="/ws"
        app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
        logger.info("✅ WebSocket路由注册完成: /ws")

        # API路由
        app.include_router(pose_router, prefix="/api/pose", tags=["姿态检测"])
        app.include_router(score_router, prefix="/api/score", tags=["评分"])

        logger.info("✅ API路由注册完成")

    except Exception as e:
        logger.error(f"❌ API路由注册失败: {e}")
        logger.error(f"请确保 api 目录存在且包含必要的文件")


# 异常处理
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "detail": str(exc)}
    )


# 主页路由
@app.get("/", response_class=HTMLResponse)
async def root():
    """主页"""
    try:
        html_path = PROJECT_ROOT / "dancevibe.html"
        if html_path.exists():
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DanceVibe API</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .container { max-width: 600px; margin: 0 auto; }
                    .logo { font-size: 3rem; margin-bottom: 20px; }
                    .status { color: #28a745; font-size: 1.2rem; margin-bottom: 30px; }
                    .links { margin-top: 30px; }
                    .links a { display: inline-block; margin: 0 10px; padding: 10px 20px; 
                             background: #007bff; color: white; text-decoration: none; 
                             border-radius: 5px; }
                    .links a:hover { background: #0056b3; }
                    .note { margin-top: 20px; color: #666; font-style: italic; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="logo">🎵 DanceVibe API</div>
                    <div class="status">✅ 服务正在运行</div>
                    <p>舞蹈姿态检测和评分系统 API 服务</p>
                    <div class="links">
                        <a href="/docs">API 文档</a>
                        <a href="/redoc">ReDoc</a>
                        <a href="/health">健康检查</a>
                    </div>
                    <div class="note">
                        🔧 已优化：WebSocket路径 /ws/ws，独立检测器实例
                    </div>
                </div>
            </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"主页加载失败: {e}")
        raise HTTPException(status_code=500, detail="页面加载失败")


# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "websocket_path": "/ws/ws",
        "features": [
            "独立检测器实例",
            "多人姿态检测", 
            "实时评分",
            "节拍提取"
        ]
    }


# WebSocket信息接口
@app.get("/ws/info")
async def websocket_info():
    """WebSocket连接信息"""
    return {
        "websocket_url": "/ws/ws",
        "protocol": "ws",
        "description": "实时舞蹈姿态检测和评分",
        "events": [
            "frame", 
            "upload_reference_video", 
            "start_game", 
            "pause_game", 
            "resume_game", 
            "stop_game"
        ]
    }


# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 DanceVibe 应用启动中...")

    # 创建必要的目录
    create_directories()

    # 设置静态文件服务
    setup_static_files()

    # 注册路由
    register_routes()

    logger.info("✅ DanceVibe 应用启动完成")
    logger.info("🔗 WebSocket 路径: /ws/ws")
    logger.info("📖 API 文档: /docs")


# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("👋 DanceVibe 应用正在关闭...")

    # 清理临时文件
    try:
        import shutil
        temp_dir = PROJECT_ROOT / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)
            logger.info("🧹 临时文件清理完成")
    except Exception as e:
        logger.error(f"❌ 临时文件清理失败: {e}")


# 命令行启动
def main():
    """命令行启动函数"""
    try:
        # 获取配置
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "true").lower() == "true"

        logger.info(f"🌐 启动服务器: http://{host}:{port}")
        logger.info(f"📖 API文档: http://{host}:{port}/docs")
        logger.info(f"🔗 WebSocket: ws://{host}:{port}/ws/ws")

        # 启动服务器
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("👋 服务器已停止")
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()