#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contains functions to render a list of ReportElement objects into various output formats
(Markdown, HTML, Word).
"""

import os
import pandas as pd
import logging
from typing import List, Dict, Any
import re

# Assuming report_elements.py is in the same directory or accessible in PYTHONPATH
try:
    from scripts.report_elements import (
        ReportElement, TitleElement, HeadingElement, ParagraphElement,
        ImageElement, TableElement, UnorderedListElement, OrderedListElement,
        RawTextElement, LineBreakElement, CodeBlockElement, TOCElement
    )
    # Placeholder for slugify, actual implementation will be in report_utils.py
    def custom_slugify_placeholder(text, separator='-'):
        text = re.sub(r'[\s()（）[\]【】:,./\\]+', separator, text) # Replace spaces and common punctuation with separator
        text = re.sub(rf'{separator}+$', '', text) # Remove trailing separators
        text = re.sub(rf'^{separator}+', '', text) # Remove leading separators
        return text.lower()

except ImportError:
    from report_elements import (
        ReportElement, TitleElement, HeadingElement, ParagraphElement,
        ImageElement, TableElement, UnorderedListElement, OrderedListElement,
        RawTextElement, LineBreakElement, CodeBlockElement, TOCElement
    )
    def custom_slugify_placeholder(text, separator='-'): # Keep consistent for standalone
        text = re.sub(r'[\s()（）[\]【】:,./\\]+', separator, text) 
        text = re.sub(rf'{separator}+$', '', text) 
        text = re.sub(rf'^{separator}+', '', text) 
        return text.lower()

logger = logging.getLogger(__name__)

# --- Markdown Renderer ---

def dataframe_to_markdown_table_renderer(df: pd.DataFrame, title: str = None, floatfmt=".2f") -> str:
    """Converts a Pandas DataFrame to a Markdown table string."""
    if df is None or df.empty:
        return ""
    
    # If index has a name, reset it to make it a column for markdown output
    df_to_render = df.copy()
    if df_to_render.index.name is not None:
        df_to_render = df_to_render.reset_index()
        
    md_table = df_to_render.to_markdown(index=False, floatfmt=floatfmt)
    if title:
        return f"#### {title}\n{md_table}\n"
    return md_table + "\n"

def _generate_toc_markdown(elements: List[ReportElement]) -> str:
    """Generates a Markdown Table of Contents from HeadingElements."""
    toc_lines = ["## 目录\n"]
    for element in elements:
        if isinstance(element, HeadingElement):
            indent = "  " * (element.level - 1)
            # Ensure element_id is generated if not present for TOC link
            element_id = element.element_id if element.element_id else custom_slugify_placeholder(element.text)
            toc_lines.append(f"{indent}- [{element.text}](#{element_id})")
    toc_lines.append("\n---\n") # Add a separator after TOC
    return "\n".join(toc_lines)

def render_elements_to_markdown(elements: List[ReportElement], output_path: str, config: Dict[str, Any]) -> None:
    """
    Renders a list of ReportElement objects to a Markdown file.

    Args:
        elements: The list of ReportElement objects.
        output_path: The path to save the generated Markdown file.
        config: Configuration dictionary (e.g., for image paths, TOC generation).
    """
    md_content_parts = []
    toc_placeholder_index = -1

    for i, element in enumerate(elements):
        if isinstance(element, TitleElement):
            md_content_parts.append(f"# {element.text}\n")
        elif isinstance(element, TOCElement):
            toc_placeholder_index = i # Mark where TOC should be inserted
            md_content_parts.append("__TOC_PLACEHOLDER__") # Temporary placeholder
        elif isinstance(element, HeadingElement):
            # Ensure element_id is generated if not explicitly set for anchor links
            if not element.element_id:
                element.element_id = custom_slugify_placeholder(element.text) # Use placeholder
            md_content_parts.append(f"{'#' * element.level} {element.text} {{#{element.element_id}}}\n")
        elif isinstance(element, ParagraphElement):
            md_content_parts.append(f"{element.text}\n")
        elif isinstance(element, ImageElement):
            # Markdown image paths are relative to the Markdown file itself.
            # The path in ImageElement should already be relative or adjusted by composer.
            alt_text = element.alt_text if element.alt_text else os.path.splitext(os.path.basename(element.path))[0]
            caption_md = f"\n*图: {element.caption}*\n" if element.caption else "\n"
            md_content_parts.append(f"![{alt_text}]({element.path}){caption_md}")
        elif isinstance(element, TableElement):
            if element.dataframe is not None:
                md_content_parts.append(dataframe_to_markdown_table_renderer(element.dataframe, title=element.title))
            elif element.markdown_string:
                table_md = f"#### {element.title}\n{element.markdown_string}\n" if element.title else element.markdown_string + "\n"
                md_content_parts.append(table_md)
        elif isinstance(element, UnorderedListElement):
            for item in element.items:
                if isinstance(item, ReportElement): # For nested elements (e.g., Paragraph in a list)
                    # This is a simplification; true nested rendering needs more context
                    # For now, just render its text representation if it's a simple element
                    if hasattr(item, 'text'): md_content_parts.append(f"- {item.text}")
                    else: md_content_parts.append(f"- [Unsupported nested element: {type(item).__name__}]")
                else: # Simple string item
                    md_content_parts.append(f"- {item}")
            md_content_parts.append("\n")
        elif isinstance(element, OrderedListElement):
            for i, item in enumerate(element.items):
                if isinstance(item, ReportElement):
                    if hasattr(item, 'text'): md_content_parts.append(f"{i+1}. {item.text}")
                    else: md_content_parts.append(f"{i+1}. [Unsupported nested element: {type(item).__name__}]")
                else:
                    md_content_parts.append(f"{i+1}. {item}")
            md_content_parts.append("\n")
        elif isinstance(element, RawTextElement):
            md_content_parts.append(element.text + "\n")
        elif isinstance(element, LineBreakElement):
            md_content_parts.append("\n---\n" * element.count if element.count > 1 else "\n") # HR for multiple
        elif isinstance(element, CodeBlockElement):
            lang = element.language if element.language else ''
            md_content_parts.append(f"```{lang}\n{element.code}\n```\n")
        else:
            logger.warning(f"Unsupported ReportElement type for Markdown: {type(element).__name__}")

    final_md_content = "\n".join(md_content_parts)

    if toc_placeholder_index != -1 and config.get("generate_toc_markdown", True):
        toc_markdown = _generate_toc_markdown(elements)
        final_md_content = final_md_content.replace("__TOC_PLACEHOLDER__", toc_markdown, 1)
    else:
        final_md_content = final_md_content.replace("__TOC_PLACEHOLDER__", "", 1) # Remove if no TOC

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_md_content)
        logger.info(f"✅ Markdown报告已保存至: {output_path}")
    except IOError as e:
        logger.error(f"❌ 保存Markdown报告失败 {output_path}: {e}")

# --- HTML Renderer (Placeholder) ---
def render_elements_to_html(elements: List[ReportElement], output_path: str, config: Dict[str, Any]) -> None:
    """
    将报告元素渲染为HTML文件
    
    Args:
        elements: 报告元素列表
        output_path: HTML文件输出路径
        config: 配置参数，包含图片目录路径等
    """
    logger.info(f"🔄 正在渲染HTML报告: {output_path}")
    
    html_parts = ["""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PWV数据分析报告</title>
        <style>
            body { font-family: 'Arial', sans-serif; margin: 20px; line-height: 1.6; color: #333; max-width: 900px; margin: auto; padding: 20px; }
            h1, h2, h3 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            h1 { text-align: center; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; border: 1px solid #ddd; }
            th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            img { max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; padding: 5px; }
            pre { background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }
            code { font-family: 'Consolas', 'Monaco', monospace; }
            .toc { background: #f9f9f9; border: 1px solid #eee; padding: 10px 20px; margin-bottom: 30px; }
            .toc ul { list-style-type: none; padding-left: 0; }
            .toc ul ul { padding-left: 20px; }
            .headerlink { font-size: 0.8em; text-decoration: none; color: #ccc; margin-left: 5px; }
            .headerlink:hover { color: #2c3e50; }
        </style>
    </head>
    <body>
    """]
    
    toc_elements = []
    toc_placeholder_index = -1
    
    for i, element in enumerate(elements):
        if isinstance(element, TitleElement):
            html_parts.append(f"<h1>{element.text}</h1>")
        elif isinstance(element, TOCElement):
            toc_placeholder_index = i
            html_parts.append("__TOC_PLACEHOLDER__")
        elif isinstance(element, HeadingElement):
            if not element.element_id:
                element.element_id = custom_slugify_placeholder(element.text)
            html_parts.append(f"<h{element.level} id=\"{element.element_id}\">{element.text}<a class=\"headerlink\" href=\"#{element.element_id}\" title=\"Permanent link\">&para;</a></h{element.level}>")
            toc_elements.append(element)  # 记录用于目录生成
        elif isinstance(element, ParagraphElement):
            html_parts.append(f"<p>{element.text}</p>")
        elif isinstance(element, ImageElement):
            # 使用配置中的图片基础路径
            image_base_dir = config.get("image_base_dir_html", "../figures")
            image_path = element.path
            
            # 修正图片路径处理逻辑：确保路径指向figures目录下的正确子目录
            if "output/figures/" in image_path:
                # 提取figures后面的相对路径，包括子目录和文件名
                rel_path = image_path.split("output/figures/")[-1]
                # 构建新路径，使用配置中的基础路径加上相对路径
                image_path = f"{image_base_dir}/{rel_path}"
            
            alt_text = element.alt_text if element.alt_text else os.path.splitext(os.path.basename(image_path))[0]
            caption = f"<figcaption>{element.caption}</figcaption>" if element.caption else ""
            html_parts.append(f"<figure><img alt=\"{alt_text}\" src=\"{image_path}\" />{caption}</figure>")
        elif isinstance(element, TableElement):
            if element.dataframe is not None:
                table_html = element.dataframe.to_html(index=False)
                if element.title:
                    html_parts.append(f"<h4>{element.title}</h4>")
                html_parts.append(table_html)
            elif element.markdown_string:
                # 简单转换markdown表格为HTML (实际应用中可能需要更复杂的处理)
                import re
                table_html = element.markdown_string.replace('\n', '<br />')
                if element.title:
                    html_parts.append(f"<h4>{element.title}</h4>")
                html_parts.append(f"<div>{table_html}</div>")
        elif isinstance(element, UnorderedListElement):
            list_items = []
            for item in element.items:
                if isinstance(item, ReportElement) and hasattr(item, 'text'):
                    list_items.append(f"<li>{item.text}</li>")
                else:
                    list_items.append(f"<li>{item}</li>")
            html_parts.append(f"<ul>{''.join(list_items)}</ul>")
        elif isinstance(element, OrderedListElement):
            list_items = []
            for item in element.items:
                if isinstance(item, ReportElement) and hasattr(item, 'text'):
                    list_items.append(f"<li>{item.text}</li>")
                else:
                    list_items.append(f"<li>{item}</li>")
            html_parts.append(f"<ol>{''.join(list_items)}</ol>")
        elif isinstance(element, RawTextElement):
            html_parts.append(f"<div>{element.text}</div>")
        elif isinstance(element, LineBreakElement):
            html_parts.append("<hr />" * element.count if element.count > 1 else "<br />")
        elif isinstance(element, CodeBlockElement):
            html_parts.append(f"<pre><code>{element.code}</code></pre>")
        else:
            logger.warning(f"不支持的HTML渲染器元素类型: {type(element).__name__}")
    
    # 生成目录
    if toc_placeholder_index != -1 and config.get("generate_toc_html", True):
        toc_html = ['<div class="toc"><h2>目录</h2><ul>']
        for element in toc_elements:
            indent = "&nbsp;" * ((element.level - 1) * 4)
            toc_html.append(f"<li>{indent}<a href=\"#{element.element_id}\">{element.text}</a></li>")
        toc_html.append("</ul></div>")
        
        html_parts[toc_placeholder_index + 1] = "".join(toc_html)
    else:
        # 如果不生成目录，移除占位符
        for i, part in enumerate(html_parts):
            if part == "__TOC_PLACEHOLDER__":
                html_parts[i] = ""
                break
    
    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"✅ HTML报告已保存至: {output_path}")
    except IOError as e:
        logger.error(f"❌ 保存HTML报告失败 {output_path}: {e}")

# --- Word Renderer (Placeholder) ---
def render_elements_to_word(elements: List[ReportElement], output_path: str, config: Dict[str, Any]) -> None:
    logger.info(f"[Placeholder] Word渲染器调用: {output_path}")
    # Actual implementation will use python-docx to build document and save
    pass

if __name__ == '__main__':
    # Dummy data for testing render_elements_to_markdown
    print("Testing report_renderers.py standalone (Markdown)..." )
    
    # Re-use or create dummy elements similar to report_composer.py test
    from scripts.report_elements import (
        TitleElement, HeadingElement, ParagraphElement, ImageElement, TableElement, TOCElement, UnorderedListElement
    )
    import pandas as pd # Make sure pandas is imported

    test_elements = [
        TitleElement("Standalone Markdown Test Report"),
        TOCElement(),
        HeadingElement(level=1, text="Section 1: Intro", element_id="section-1-intro"),
        ParagraphElement("This is a test paragraph for section 1."),
        ImageElement(path="output/figures/dummy_image.png", caption="Dummy Image for Test"),
        HeadingElement(level=2, text="Subsection 1.1", element_id="subsection-1-1"),
        UnorderedListElement(["Point A", "Point B"]),
        TableElement(title="Test Table", dataframe=pd.DataFrame({'A': [1,2], 'B': [3,4]}))
    ]
    test_config = {
        "generate_toc_markdown": True,
        "image_base_dir_markdown": "output/figures" # Example config for image paths
    }
    output_md_path = "output/reports/test_standalone_render.md"

    render_elements_to_markdown(test_elements, output_md_path, test_config)
    print(f"Markdown rendering test complete. Check: {output_md_path}")

    # You would add similar tests for HTML and Word renderers once implemented 