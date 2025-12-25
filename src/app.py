"""
Gradio-based Web Interface for Image Retrieval System
Provides interactive image search functionality
"""
import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_TOP_K, INDEX_DIR
from retrieval_engine import RetrievalEngine


# Global engine instance
engine: RetrievalEngine = None


def load_engine():
    """Load the retrieval engine with pre-built index"""
    global engine
    if engine is None:
        print("Loading retrieval engine...")
        engine = RetrievalEngine()
        try:
            engine.load(INDEX_DIR)
            print(f"Engine loaded with {len(engine.image_paths)} indexed images")
        except FileNotFoundError:
            print("No pre-built index found. Please run build_index.py first.")
            raise gr.Error("索引文件未找到，请先运行 scripts/build_index.py 构建索引")
    return engine


def search_by_image(query_image: Image.Image, top_k: int = 10) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """
    Search for similar images given a query image
    
    Returns:
        gallery: List of (image, caption) tuples
        info: Search info text
    """
    if query_image is None:
        return [], "请上传一张图片"
    
    engine = load_engine()
    
    # Perform search
    results = engine.search(query_image, k=int(top_k))
    
    # Prepare gallery items
    gallery_items = []
    for r in results:
        try:
            img = Image.open(r["path"]).convert("RGB")
            # Resize for display
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            caption = f"#{r['rank']} {r['class_name']}\n距离: {r['distance']:.4f}"
            gallery_items.append((img, caption))
        except Exception as e:
            print(f"Error loading image {r['path']}: {e}")
            
    info = f"找到 {len(results)} 个相似图像 (Top-{top_k})"
    
    return gallery_items, info


def search_by_text(query_text: str, top_k: int = 10) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """
    Search for images by text description
    
    Returns:
        gallery: List of (image, caption) tuples
        info: Search info text
    """
    if not query_text or not query_text.strip():
        return [], "请输入搜索文本"
    
    engine = load_engine()
    
    # Perform text search
    results = engine.search_by_text(query_text.strip(), k=int(top_k))
    
    # Prepare gallery items
    gallery_items = []
    for r in results:
        try:
            img = Image.open(r["path"]).convert("RGB")
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            caption = f"#{r['rank']} {r['class_name']}\n距离: {r['distance']:.4f}"
            gallery_items.append((img, caption))
        except Exception as e:
            print(f"Error loading image {r['path']}: {e}")
            
    info = f"文本搜索: '{query_text}' - 找到 {len(results)} 个结果"
    
    return gallery_items, info


def create_ui() -> gr.Blocks:
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="图像检索系统",
        theme=gr.themes.Soft(),
        css="""
        .gallery-item { border-radius: 8px; }
        .search-box { max-width: 600px; margin: auto; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🔍 图像检索系统
            ### 基于CLIP+FAISS的高效图像检索
            
            上传一张图片或输入文本描述，系统将返回最相似的图像。
            """
        )
        
        with gr.Tabs():
            # Tab 1: Image Search
            with gr.TabItem("📷 以图搜图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_image = gr.Image(
                            label="上传查询图片",
                            type="pil",
                            height=300
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=DEFAULT_TOP_K,
                            step=1,
                            label="返回结果数量"
                        )
                        search_btn = gr.Button("🔍 搜索相似图片", variant="primary")
                        
                    with gr.Column(scale=2):
                        result_info = gr.Textbox(label="搜索信息", interactive=False)
                        result_gallery = gr.Gallery(
                            label="检索结果",
                            columns=5,
                            rows=2,
                            height="auto",
                            object_fit="cover"
                        )
                        
                search_btn.click(
                    fn=search_by_image,
                    inputs=[query_image, top_k_slider],
                    outputs=[result_gallery, result_info]
                )
                
            # Tab 2: Text Search
            with gr.TabItem("📝 文本搜索"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_text = gr.Textbox(
                            label="输入搜索文本",
                            placeholder="例如: a red car, a cute dog, sunset over ocean...",
                            lines=3
                        )
                        top_k_text = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=DEFAULT_TOP_K,
                            step=1,
                            label="返回结果数量"
                        )
                        text_search_btn = gr.Button("🔍 搜索", variant="primary")
                        
                    with gr.Column(scale=2):
                        text_result_info = gr.Textbox(label="搜索信息", interactive=False)
                        text_result_gallery = gr.Gallery(
                            label="检索结果",
                            columns=5,
                            rows=2,
                            height="auto",
                            object_fit="cover"
                        )
                        
                text_search_btn.click(
                    fn=search_by_text,
                    inputs=[query_text, top_k_text],
                    outputs=[text_result_gallery, text_result_info]
                )
                
            # Tab 3: About
            with gr.TabItem("ℹ️ 关于"):
                gr.Markdown(
                    """
                    ## 系统介绍
                    
                    本系统是一个基于深度学习的图像检索系统，具有以下特点：
                    
                    ### 技术架构
                    - **特征提取**: 使用OpenAI CLIP (ViT-B/32)模型提取512维图像特征
                    - **向量检索**: 使用FAISS进行高效的相似度搜索，支持GPU加速
                    - **数据集**: CIFAR-10 (60,000张图像，10个类别)
                    
                    ### 功能特点
                    - ✅ 以图搜图：上传图片查找相似图像
                    - ✅ 文本搜索：通过文字描述搜索图像(CLIP跨模态能力)
                    - ✅ 实时检索：毫秒级响应速度
                    - ✅ GPU加速：支持CUDA加速的特征提取和检索
                    
                    ### 使用说明
                    1. 在"以图搜图"标签页上传一张图片
                    2. 调整返回结果数量
                    3. 点击搜索按钮查看结果
                    
                    ---
                    *Powered by CLIP + FAISS*
                    """
                )
                
    return demo


def main():
    """Main entry point"""
    demo = create_ui()
    
    # Pre-load engine
    try:
        load_engine()
    except Exception as e:
        print(f"Warning: Could not pre-load engine: {e}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
