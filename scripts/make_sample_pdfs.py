"""生成 sample_papers/ 下的示例 PDF（开发与验证用）。

用法：.venv/Scripts/python scripts/make_sample_pdfs.py
"""
from pathlib import Path

import fitz

OUT = Path(__file__).resolve().parent.parent / "sample_papers"

ZH_TITLE = "基于注意力机制的文本分类方法研究"
EN_TITLE = "Attention Mechanisms in Sequence Modeling"


def make_pdf(path: Path, title: str, author: str, year: int, body_pages: list[str]) -> None:
    doc = fitz.open()
    for text in body_pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, text, fontname="china-s", fontsize=11)
    doc.set_metadata(
        {"title": title, "author": author, "creationDate": f"D:{year}0101000000"}
    )
    doc.save(str(path))
    doc.close()


def _zh_pages() -> list[str]:
    paras = [
        "文本分类是自然语言处理领域的经典任务，目标是将文本划分到预定义的类别中。"
        "传统方法依赖人工设计的特征，如词袋模型和 TF-IDF 统计特征，再配合支持向量机等分类器。"
        "随着深度学习的兴起，卷积神经网络与循环神经网络被广泛用于文本分类，取得了显著进展。",
        "注意力机制最初被提出用于机器翻译任务，其核心思想是让模型在生成每个输出时，"
        "动态地关注输入序列中更重要的部分。注意力机制可以缓解长序列中信息丢失的问题，"
        "显著提升模型对长距离依赖的建模能力。",
        "Transformer 架构完全基于自注意力机制构建，摒弃了循环结构，支持并行计算。"
        "其多头注意力模块从多个表示子空间捕获不同的语义关系，位置编码则为模型注入序列顺序信息。"
        "基于 Transformer 的预训练语言模型在各类文本分类基准上大幅刷新了纪录。",
        "本文提出的方法将多头注意力与文本分类任务结合，在情感分析、主题分类等数据集上进行实验。"
        "实验结果表明，所提方法相较基线模型在准确率和 F1 值上均有明显提升，"
        "消融实验也验证了注意力模块对性能的贡献。",
    ]
    return [paras[0] + paras[1], paras[2] + paras[3]]


def _en_pages() -> list[str]:
    paras = [
        "Sequence modeling is a fundamental problem in natural language processing. "
        "Recurrent neural networks have long been the dominant approach, but their sequential "
        "nature limits parallelization and makes it hard to capture long-range dependencies. "
        "Convolutional approaches offer parallelism but require many layers to reach long distances.",
        "Attention mechanisms allow a model to focus on relevant parts of the input when "
        "producing each output element. The Transformer architecture extends this idea by "
        "relying entirely on self-attention, with multi-head attention projecting queries, "
        "keys and values into multiple subspaces. Positional encodings inject information "
        "about the order of tokens into the otherwise order-invariant attention computation.",
        "This paper studies how attention-based architectures behave on sequence classification "
        "tasks. We compare Transformer models with recurrent baselines on several benchmark "
        "datasets and analyze attention weights to understand which tokens the model relies on. "
        "The results show that attention-based models achieve competitive accuracy while "
        "training substantially faster on modern hardware.",
    ]
    return [paras[0] + paras[1], paras[2]]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    make_pdf(OUT / "zh_paper.pdf", ZH_TITLE, "张三", 2023, _zh_pages())
    make_pdf(OUT / "en_paper.pdf", EN_TITLE, "Alice Wang", 2020, _en_pages())
    print(f"已生成 {len(list(OUT.glob('*.pdf')))} 个示例 PDF 到 {OUT}")


if __name__ == "__main__":
    main()
