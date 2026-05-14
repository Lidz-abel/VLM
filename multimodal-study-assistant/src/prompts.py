IMAGE_ANALYSIS_PROMPT = """你是一个面向学习资料的多模态解析助手。
请根据图片内容进行结构化分析，必须使用 Markdown，并严格包含以下栏目：

【图片类型】
从以下类型中选择一个并说明理由：
- exam_question：考试题 / 作业题
- paper_figure：论文图表
- code_screenshot：代码或报错截图
- slide：课件截图
- chart：统计图表
- other：其他

【识别内容】
尽可能准确转写图片中的文字、公式、代码、表格标题、坐标轴、图例和题干信息。
如果有看不清的内容，请明确标注“无法确认”，不要编造。

【核心知识点】
用条目列出本页或图片涉及的关键概念。

【分析过程】
结合图片内容逐步解释。若是题目，请给出解题思路；若是图表，请解释变量、趋势和结论；若是代码截图，请定位可能的错误原因。

【最终答案】
给出面向用户问题的直接回答。如果无法确定，请说明缺失的信息。
"""

IMAGE_QA_PROMPT = """请基于图片内容回答用户问题。回答要准确、简洁，并说明依据来自图片中的哪些信息。"""

PDF_PAGE_ANALYSIS_PROMPT = """你正在解析一份 PDF 的单页图片。请输出稳定、可用于检索的 Markdown 内容。

必须严格包含以下栏目：

【本页主题】
概括本页主题。

【OCR 转写内容】
尽可能准确转写本页文字、公式、表格标题、坐标轴、图例、代码和题目内容。

【关键概念】
用条目列出本页涉及的核心概念。

【图表/公式/代码说明】
解释本页出现的图表、公式或代码；如果没有，请写“无明显图表/公式/代码”。

【本页摘要】
用 3-6 句话总结本页内容，适合后续检索。

如果图片内容不完整或无法辨认，请如实说明，不要补全不存在的信息。
"""

NOTE_GENERATION_PROMPT = """请把下面的学习资料解析内容整理成 Markdown 复习笔记。

要求：
- 结构清晰，适合复习
- 保留重要公式、定义、图表结论和题目思路
- 不要编造资料中没有的信息
- 使用以下章节：

# 主题名称

## 1. 核心概念

## 2. 主要内容

## 3. 公式 / 图表解释

## 4. 典型题目

## 5. 易错点

## 6. 总结
"""

REVIEW_QUESTION_PROMPT = """请根据下面的学习资料生成复习题。
要求包含选择题、简答题和应用题，并给出参考答案。不要生成资料中没有依据的题目。"""


def build_image_prompt(user_question: str | None = None, structured: bool = True) -> str:
    base = IMAGE_ANALYSIS_PROMPT if structured else IMAGE_QA_PROMPT
    if user_question and user_question.strip():
        return f"{base}\n\n用户额外问题：{user_question.strip()}"
    return base


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"【第 {item.get('page', '?')} 页】\n{item.get('content', '')}"
        for item in contexts
    )
    return f"""你是一个严谨的学习资料问答助手。请只基于给定 PDF 页面解析内容回答问题。

如果上下文中没有答案，请明确说“当前检索到的页面不足以回答”，不要编造。
回答中必须说明依据来自哪些页。

【用户问题】
{question}

【检索到的页面内容】
{context_text}
"""
