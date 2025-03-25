[TableRAG: Million-Token Table Understanding with Language Models](https://yiyibooks.cn/arxiv/2410.04739v3/index.html)
https://arxiv.org/abs/2410.04739
https://github.com/google-research/google-research/tree/master/table_rag

这篇论文是谷歌2024年12月发布的paper，针对的是百万级别的超大的表格搜索，但是里面提到了很多表格检索的方法，还是很值得借鉴的。

![[TableRAG.png]]
文章提出了一个观点：**模式检索和单元格检索的效果是最好的，而且综合这两种检索方式的结果能够达到最高效率**。

整个过程中，算法不是核心，看看框架的数据处理的具体代码实现：

```python
# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, List, Any
from collections import Counter

import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings


class Retriever:
    def __init__(self, agent_type, mode, embed_model_name, top_k = 5, max_encode_cell = 10000, db_dir = 'db/', verbose = False):
        self.agent_type = agent_type
        self.mode = mode
        self.embed_model_name = embed_model_name
        self.schema_retriever = None
        self.cell_retriever = None
        self.row_retriever = None
        self.column_retriever = None
        self.top_k = top_k
        self.max_encode_cell = max_encode_cell
        self.db_dir = db_dir
        self.verbose = verbose
        os.makedirs(db_dir, exist_ok=True)

        if self.mode == 'bm25':
            self.embedder = None
        elif 'text-embedding' in self.embed_model_name:
            self.embedder = OpenAIEmbeddings(model=self.embed_model_name)
        elif 'gecko' in self.embed_model_name: # VertexAI
            self.embedder = VertexAIEmbeddings(model_name=self.embed_model_name)
        else:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)

    def init_retriever(self, table_id, df):
        self.df = df
        if 'TableRAG' in self.agent_type:
            self.schema_retriever = self.get_retriever('schema', table_id, self.df)
            self.cell_retriever = self.get_retriever('cell', table_id, self.df)
        elif self.agent_type == 'TableSampling':
            max_row = max(1, self.max_encode_cell // 2 // len(self.df.columns))
            self.df = self.df.iloc[:max_row]
            self.row_retriever = self.get_retriever('row', table_id, self.df)
            self.column_retriever = self.get_retriever('column', table_id, self.df)

    def get_retriever(self, data_type, table_id, df):
        docs = None
        if self.mode == 'embed' or self.mode == 'hybrid':
            db_dir = os.path.join(self.db_dir, f'{data_type}_db_{self.max_encode_cell}_' + table_id)
            if os.path.exists(db_dir):
                if self.verbose:
                    print(f'Load {data_type} database from {db_dir}')
                db = FAISS.load_local(db_dir, self.embedder, allow_dangerous_deserialization=True)
            else:
                docs = self.get_docs(data_type, df)
                db = FAISS.from_documents(docs, self.embedder)
                db.save_local(db_dir)
            embed_retriever = db.as_retriever(search_kwargs={'k': self.top_k})
        if self.mode == 'bm25' or self.mode == 'hybrid':
            if docs is None:
                docs = self.get_docs(data_type, df)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = self.top_k
        if self.mode == 'hybrid':
            # return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.9, 0.1])
            return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.5, 0.5])
        elif self.mode == 'embed':
            return embed_retriever
        elif self.mode == 'bm25':
            return bm25_retriever

    def get_docs(self, data_type, df):
        if data_type == 'schema':
            return self.build_schema_corpus(df)
        elif data_type == 'cell':
            return self.build_cell_corpus(df)
        elif data_type == 'row':
            return self.build_row_corpus(df)
        elif data_type == 'column':
            return self.build_column_corpus(df)

    def build_schema_corpus(self, df):
        docs = []
        for col_name, col in df.items():
            if col.dtype != 'object' and col.dtype != str:
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}'
            else:
                most_freq_vals = col.value_counts().index.tolist()
                example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "cell_examples": {example_cells}}}'
            docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
        return docs

    def build_cell_corpus(self, df):
        docs = []
        categorical_columns = df.columns[(df.dtypes == 'object') | (df.dtypes == str)]
        other_columns = df.columns[~(df.dtypes == 'object') | (df.dtypes == str)]
        if len(other_columns) > 0:
            for col_name in other_columns:
                col = df[col_name]
                docs.append(f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}')
        if len(categorical_columns) > 0:
            cell_cnt = Counter(df[categorical_columns].apply(lambda x: '{"column_name": "' + x.name + '", "cell_value": "' + x.astype(str) + '"}').values.flatten())
            docs += [cell for cell, _ in cell_cnt.most_common(self.max_encode_cell - len(docs))]
        docs = [Document(page_content=doc) for doc in docs]
        return docs

    def build_row_corpus(self, df):
        row_docs = []
        for row_id, (_, row) in enumerate(df.iterrows()):
            row_text = '|'.join(str(cell) for cell in row)
            row_doc = Document(page_content=row_text, metadata={'row_id': row_id})
            row_docs.append(row_doc)
        return row_docs

    def build_column_corpus(self, df):
        col_docs = []
        for col_id, (_, column) in enumerate(df.items()):
            col_text = '|'.join(str(cell) for cell in column)
            col_doc = Document(page_content=col_text, metadata={'col_id': col_id})
            col_docs.append(col_doc)
        return col_docs

    def retrieve_schema(self, query):
        results = self.schema_retriever.invoke(query)
        observations = [doc.metadata['result_text'] for doc in results]
        return observations

    def retrieve_cell(self, query):
        results = self.cell_retriever.invoke(query)
        observations = [doc.page_content for doc in results]
        return observations

    def sample_rows_and_columns(self, query):
        # Apply row sampling
        row_results = self.row_retriever.invoke(query)
        row_ids = sorted([doc.metadata['row_id'] for doc in row_results])
        # Apply column sampling
        col_results = self.column_retriever.invoke(query)
        col_ids = sorted([doc.metadata['col_id'] for doc in col_results])
        # Return sampled rows and columns
        return self.df.iloc[row_ids, col_ids]
```

### 整体代码功能

`Retriever` 类是一个表格检索器，支持两种主要模式：

1. **TableRAG**: 针对表格的模式，分为 schema（模式/结构）和 cell（单元格）两种检索方式。
2. **TableSampling**: 针对行列采样的模式，分为 row（行）和 column（列）两种检索方式。

它支持三种检索方法：

- **BM25**: 基于关键词的传统检索。
- **Embed**: 基于向量嵌入的语义检索。
- **Hybrid**: 结合 BM25 和嵌入的混合检索。

核心功能是通过 `build_*` 函数将表格数据（`pandas.DataFrame`）转换为 `Document` 对象列表，然后利用这些文档构建检索器（FAISS 或 BM25）。

---

### 关键函数分析：`build_*` 函数

`build_*` 函数的作用是将表格数据按不同粒度（schema、cell、row、column）转换为检索用的文档格式。以下是每个函数的详细分析：

#### 1. `build_schema_corpus(self, df)`

**作用**: 将表格的模式信息（列名和列的统计信息）转换为文档，用于检索与表格结构相关的查询。

**实现**:
- 输入是一个 `pandas.DataFrame`（`df`）。
- 对每一列（`col_name` 和 `col`）进行处理：
  - 如果列的数据类型不是字符串（`object` 或 `str`）：
    - 生成一个 JSON 字符串，包含列名、数据类型、最小值和最大值。
    - 示例：`{"column_name": "age", "dtype": "int64", "min": 18, "max": 65}`
  - 如果列是字符串类型：
    - 统计列中最常见的值（`value_counts`），取前 3 个作为示例值。
    - 生成一个 JSON 字符串，包含列名、数据类型和示例值。
    - 示例：`{"column_name": "city", "dtype": "object", "cell_examples": ["New York", "London", "Paris"]}`
- 每个列生成一个 `Document` 对象：
  - `page_content` 是列名（用于检索时的匹配）。
  - `metadata` 中存储完整的 JSON 字符串（用于返回结果）。

**意义**: 
- 这种格式化允许检索器快速找到与表格结构相关的列（例如查询“哪一列有年龄信息”）。
- 通过区分数值和字符串类型，提供更有针对性的描述。

---

#### 2. `build_cell_corpus(self, df)`

**作用**: 将表格的单元格信息转换为文档，用于检索具体的单元格值。

**实现**:
- 将列分为两类：
  - **非字符串列**（数值型等）：为每列生成一个 JSON 字符串，包含列名、数据类型、最小值和最大值。
  - **字符串列**（`object` 或 `str`）：统计所有单元格值的频率（使用 `Counter`），并取最常见的 `max_encode_cell - 数值列数量` 个单元格值。
- 每个文档是一个 `Document` 对象，`page_content` 是 JSON 字符串。
- 示例：
  - 数值列：`{"column_name": "price", "dtype": "float64", "min": 10.5, "max": 99.9}`
  - 字符串列：`{"column_name": "color", "cell_value": "red"}`

**意义**:
- 专注于单元格级别的信息，适合回答“表格中有没有某个值”之类的问题。
- 对字符串列使用频率统计，确保高频值被优先编码，控制文档总数不超过 `max_encode_cell`。

---

#### 3. `build_row_corpus(self, df)`

**作用**: 将表格的每一行转换为文档，用于行级别的检索。

**实现**:
- 遍历 `df` 的每一行（使用 `iterrows`）。
- 将一行中的所有单元格值用 `|` 分隔，生成字符串。
  - 示例：如果一行是 `[1, "apple", 3.5]`，则 `page_content` 为 `"1|apple|3.5"`。
- 创建 `Document` 对象：
  - `page_content` 是行内容的字符串。
  - `metadata` 中存储行号（`row_id`）。

**意义**:
- 适合基于行内容的语义或关键词检索，例如“找包含 apple 的行”。
- 行号存储在 metadata 中，便于后续定位具体行。

---

#### 4. `build_column_corpus(self, df)`

**作用**: 将表格的每一列转换为文档，用于列级别的检索。

**实现**:
- 遍历 `df` 的每一列（使用 `items`）。
- 将一列中的所有单元格值用 `|` 分隔，生成字符串。
  - 示例：如果一列是 `[1, 2, 3]`，则 `page_content` 为 `"1|2|3"`。
- 创建 `Document` 对象：
  - `page_content` 是列内容的字符串。
  - `metadata` 中存储列号（`col_id`）。

**意义**:
- 适合基于列内容的检索，例如“找包含特定值的列”。
- 列号存储在 metadata 中，便于后续定位具体列。

---

### 这些函数的作用总结

1. **`build_schema_corpus`**: 提供表格的结构化信息（列名、类型、统计值），用于回答与表格模式相关的问题。
2. **`build_cell_corpus`**: 提供单元格级别的细节，适合精确查找具体值。
3. **`build_row_corpus`**: 将行作为整体编码，适合基于行内容的检索。
4. **`build_column_corpus`**: 将列作为整体编码，适合基于列内容的检索。

这些函数的输出（`Document` 列表）会被传入 `get_retriever` 函数，用于构建 FAISS（向量检索）或 BM25（关键词检索）的索引。