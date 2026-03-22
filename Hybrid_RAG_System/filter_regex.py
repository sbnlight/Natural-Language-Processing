import os
import json
import re
import shutil  # 引入 shutil 用于移动淘汰的文件

def clean_content_regex():
    # 输入文件夹（也是最终干净文件保留的文件夹）
    input_dir = "./corpus/filtered_texts_big"
    
    # 本次清洗被淘汰文件存放的专属文件夹
    rejected_dir = "./corpus/filtered_texts_big_regex_rejected"
    
    if not os.path.exists(input_dir):
        print(f"找不到输入目录: {input_dir}")
        return

    os.makedirs(rejected_dir, exist_ok=True)

    # ==========================================
    # Chunk 级别过滤规则（正则表达式）
    # 只要 Chunk 命中以下任意规则，该 Chunk 就会被删除
    # ==========================================
    bad_chunk_patterns =[
        r'Size\s*\(bytes\)',          # 命中示例: Size (bytes)
        r'Last\s*Modified',           # 命中示例: Last Modified
        r'Filename\s*Readable\?',     # 命中示例: Filename Readable?
        r'Index of\s+/',              # 命中典型的 Apache 目录索引头部: Index of /
        r'Parent Directory',          # 命中返回上一级目录的链接文本: Parent Directory
        r'Apache/.*?Server at',       # 命中底部的服务器签名: Apache/2.4.41 Server at ... Port 443
        r'Proudly served by lighthttpd', # 其他服务器签名
    ]
    
    # 编译正则
    compiled_patterns =[re.compile(p, re.IGNORECASE) for p in bad_chunk_patterns]

    total_files = 0
    rejected_files = 0
    kept_files = 0
    total_chunks_before = 0
    total_chunks_removed = 0

    print("开始进行内容正则清洗...")

    # 先获取所有 json 文件的列表，防止边修改边遍历导致问题
    json_files =[f for f in os.listdir(input_dir) if f.endswith(".json")]

    for filename in json_files:
        total_files += 1
        filepath = os.path.join(input_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_chunks = data.get("chunks", [])
            total_chunks_before += len(original_chunks)
            
            valid_chunks =[]
            
            # 遍历检查每一个 chunk
            for chunk in original_chunks:
                text = chunk.get("text", "")
                is_bad_chunk = False
                
                # 检查是否命中正则黑名单
                for pattern in compiled_patterns:
                    if pattern.search(text):
                        is_bad_chunk = True
                        break
                
                # 检查字符比例
                letters = sum(c.isalpha() for c in text)
                if len(text) > 50 and (letters / len(text)) < 0.4:
                    is_bad_chunk = True

                if not is_bad_chunk:
                    valid_chunks.append(chunk)
                else:
                    total_chunks_removed += 1

            # 评估清洗后的文件整体质量
            total_valid_text_length = sum(len(c.get("text", "")) for c in valid_chunks)
            
            if len(valid_chunks) == 0 or total_valid_text_length < 50:
                # ==========================
                # 文件被淘汰：直接从原文件夹移走
                # ==========================
                rejected_files += 1
                shutil.move(filepath, os.path.join(rejected_dir, filename))
            else:
                # ==========================
                # 文件被保留：将剔除垃圾后的干净内容覆写回原文件
                # ==========================
                kept_files += 1
                data["chunks"] = valid_chunks
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"处理文件出错 {filename}: {e}")

        # 进度提示
        if total_files % 1000 == 0:
            print(f"已处理 {total_files} 个文件...")

    # 打印统计信息
    print("\n" + "="*40)
    print("第二遍（正则内容）清洗完成！统计信息如下：")
    print(f"处理文件总数:       {total_files}")
    print(f"完全淘汰文件数:     {rejected_files} -> 已移至 {rejected_dir}")
    print(f"成功保留文件数:     {kept_files} -> 保留在原处 {input_dir}")
    print("-" * 40)
    if total_chunks_before > 0:
        print(f"处理 Chunk 总数:    {total_chunks_before}")
        print(f"剔除垃圾 Chunk 数:  {total_chunks_removed} ({(total_chunks_removed/total_chunks_before)*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    clean_content_regex()