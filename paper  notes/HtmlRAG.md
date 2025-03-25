https://arxiv.org/abs/2411.02959
https://github.com/plageon/HtmlRAG

## 核心

这是一篇难得的好文章，充分谈及了目前RAG中应对HTML的处理方式的弊端以及改进，创新性提出了完全基于“树”的RAG运用范式。其中，最为关键的在于细粒度控制和两步基于块树的剪枝。***No chunk based search but fully pruned!***

![[html rag method.png]]

## **Two-Step Block-Tree-Based HTML Pruning**

第一步使用了Embedding模型的相似度计算剪枝。这个很简单，将大的block的分数较低的排除了。第二步是重点，在我的理解中，其实第二步优化的是第一步得到的block tree，这个block tree有在此步骤中进行扩展。然后使用GPT模型在这棵树上进行一边DFS一边计算分数，这里的分数计算就很取巧，利用的是Base Model打榜的做法，将从根节点下来的路径节点序列作为输入，然后将输出的Logits来作为计算分数的依据。

![[html rag token probability calculation.png]]

详细的代码实现（序列其实包含了template的内容）：

```python
def generate_html_tree(self,
                       tokenizer,
                       query: List[str],
                       htmls: List[List[str]],
                       block_tree: List[Tuple],
                       **kwargs):
    # 设置最大序列长度，默认为131072
    max_seq_length = kwargs.pop("max_seq_length", 131072)

    # 内部函数：应用HTML树模板
    def apply_html_tree_template(query, htmls):
        template = """**HTML**: ```{input_html}```\n**Question**: **{question}**\n Your task is to identify the most relevant text piece to the given question in the HTML document. This text piece could either be a direct paraphrase to the fact, or a supporting evidence that can be used to infer the fact. The overall length of the text piece should be more than 300 words and less than 500 words. You should provide the path to the text piece in the HTML document. An example for the output is: <html 1><body><div 2><p>Some key information..."""
        return template.format(input_html="\n".join(htmls), question=query)

    res_html_refs = []
    # 遍历每个HTML文档
    for idx, _htmls in enumerate(htmls):
        # 处理HTML长度，确保不超过最大序列长度
        if isinstance(_htmls, str):
            _htmls = [_htmls]
        else:
            html_token_lens = [len(tokenizer.encode(html)) for html in _htmls]
            total_html_token_len = sum(html_token_lens)
            while total_html_token_len > max_seq_length - 2048:
                if len(_htmls) == 1:
                    break
                max_length_idx = html_token_lens.index(max(html_token_lens))
                html_token_lens.pop(max_length_idx)
                _htmls.pop(max_length_idx)
                total_html_token_len = sum(html_token_lens)

        # 准备模型输入
        model_input = apply_html_tree_template(query, _htmls)
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": model_input}], 
            add_special_tokens=True,
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt",
            return_dict=True
        )

        # 使用BeautifulSoup解析HTML
        soup = bs4.BeautifulSoup("", 'html.parser')
        for html in _htmls:
            soup.append(bs4.BeautifulSoup(html, 'html.parser'))

        # 处理块树和路径
        token_id_paths = []
        _block_tree = block_tree[idx]
        is_leaf = [p[2] for p in _block_tree]
        _block_tree = [p[1] for p in _block_tree]

        for path in _block_tree:
            path_str = "<" + "><".join(path) + ">"
            token_ids = tokenizer.encode(path_str, add_special_tokens=False)
            token_id_paths.append(token_ids)

        # 构建TokenId树
        root = TokenIdNode(-1)
        for path in token_id_paths:
            parent = root
            for i, token_id in enumerate(path):
                has_child = False
                for child in parent.children:
                    if child.name == token_id:
                        parent = child
                        has_child = True
                        break
                if not has_child:
                    node = TokenIdNode(token_id, parent=parent, input_ids=path[:i + 1])
                    parent = node

        # 计算节点转换概率
        node_queue = [root]
        while node_queue:
            cur_node = node_queue.pop(0)
            children = cur_node.children
            
            # 单个子节点处理
            if len(children) == 1:
                cur_node.children[0].prob = str(np.float32(1.0))
                node_queue.append(children[0])
                continue
            elif len(children) == 0:
                continue

            # 多子节点概率计算
            force_token_id = [c.name for c in children]
            child_input_ids = torch.tensor(cur_node.input_ids, dtype=torch.long).unsqueeze(0)
            child_input_ids = torch.cat([inputs["input_ids"][idx:idx + 1], child_input_ids], dim=1).to(self.device)
            
            model_inputs = self.prepare_inputs_for_generation(child_input_ids, **kwargs)
            outputs = self(**model_inputs, return_dict=True)
            
            force_token_id = torch.tensor(force_token_id, device=self.device)
            probs = torch.gather(outputs.logits[:, 0, :], -1, force_token_id.unsqueeze(0))
            probs = torch.nn.functional.softmax(probs, dim=-1)
            probs = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()
            
            # 更新每个子节点的概率
            for i, child in enumerate(children):
                child.prob = str(probs[i])
                node_queue.append(child)

        # 收集结果
        res_html_refs.append({
            "html": str(soup),
            "paths": _block_tree,
            "is_leaf": is_leaf,
            "path_token_ids": token_id_paths,
            "node_tree": list(TokenDotExporter(root, nodenamefunc=nodenamefunc))
        })
    
    return res_html_refs
```




