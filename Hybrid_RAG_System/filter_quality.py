import os
import json
import shutil
import hashlib

def clean_content_quality_dedup():
    # 输入文件夹（也是最终干净文件保留的文件夹）
    input_dir = "./corpus/filtered_texts_big"
    
    # 本次清洗被淘汰文件存放的专属文件夹
    rejected_dir = "./corpus/filtered_texts_big_quality_rejected"
    
    if not os.path.exists(input_dir):
        print(f"找不到输入目录: {input_dir}")
        return

    os.makedirs(rejected_dir, exist_ok=True)

    total_files = 0
    rejected_files = 0
    kept_files = 0
    total_chunks_before = 0
    total_chunks_removed = 0
    
    # 用于全局去重的 Hash 集合
    seen_chunk_hashes = set()
    
    # 质量控制参数
    MIN_WORDS_IN_CHUNK = 5      # 一个 chunk 最少需要包含几个单词（按空格分割）
    MIN_VALID_TEXT_LENGTH = 50  # 整个网页剩下的有效字符总长度少于此值，则淘汰该网页

    print("开始进行第三遍清洗 (去重与语义质量)...")

    # 先获取所有 json 文件的列表，防止边修改边遍历导致问题
    json_files =[f for f in os.listdir(input_dir) if f.endswith(".json")]

    for filename in json_files:
        total_files += 1
        filepath = os.path.join(input_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_chunks = data.get("chunks",[])
            total_chunks_before += len(original_chunks)
            
            valid_chunks =[]
            
            # 遍历检查每一个 chunk
            for chunk in original_chunks:
                text = chunk.get("text", "").strip()
                is_bad_chunk = False
                
                # 规则 1：语义长度过滤 (过滤碎片化文本)
                # 计算有效单词数（按空白字符分割）
                words = text.split()
                if len(words) < MIN_WORDS_IN_CHUNK:
                    is_bad_chunk = True
                
                # 规则 2：全局精确去重 (过滤全站公共导航、版权申明等)
                # 使用 MD5 计算文本的哈希值，节省内存且比原生 hash() 稳定
                if not is_bad_chunk:
                    chunk_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    if chunk_hash in seen_chunk_hashes:
                        # 发现重复的公共模板文本，丢弃
                        is_bad_chunk = True
                    else:
                        # 第一次遇见的文本，记录它的哈希值
                        seen_chunk_hashes.add(chunk_hash)

                if not is_bad_chunk:
                    valid_chunks.append(chunk)
                else:
                    total_chunks_removed += 1

            # 评估清洗后的文件整体质量
            total_valid_text_length = sum(len(c.get("text", "")) for c in valid_chunks)
            
            # 如果剔除垃圾 chunk 后，文件为空，或者剩下的文本太短
            if len(valid_chunks) == 0 or total_valid_text_length < MIN_VALID_TEXT_LENGTH:
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
    print("\n" + "="*45)
    print("第三遍（去重与质量）清洗完成！统计信息如下：")
    print(f"处理文件总数:       {total_files}")
    print(f"完全淘汰文件数:     {rejected_files} -> 已移至 {rejected_dir}")
    print(f"成功保留文件数:     {kept_files} -> 保留在原处 {input_dir}")
    print("-" * 45)
    if total_chunks_before > 0:
        print(f"处理 Chunk 总数:    {total_chunks_before}")
        print(f"剔除冗余/过短 Chunk: {total_chunks_removed} ({(total_chunks_removed/total_chunks_before)*100:.2f}%)")
        print(f"全局唯一有效 Chunk: {len(seen_chunk_hashes)}")
    print("="*45)

if __name__ == "__main__":
    clean_content_quality_dedup()