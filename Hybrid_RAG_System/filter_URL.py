import os
import json
import shutil
import re

def clean_urls():
    # 定义输入和输出文件夹
    input_dir = "./corpus/filtered_texts_big"
    rejected_dir = "./corpus/filtered_texts_big_URL_rejected"
    
    # 确保输入文件夹存在
    if not os.path.exists(input_dir):
        print(f"找不到输入目录: {input_dir}")
        return

    # 创建用于存放被剔除文件的文件夹
    os.makedirs(rejected_dir, exist_ok=True)

    # ==========================================
    # URL 过滤规则（正则表达式）
    # ==========================================
    unwanted_patterns =[
        # 1. 动态脚本和查询接口（绝大多数是自动生成的目录或数据库查询）
        r'/cgi-bin/',
        r'\?file=',
        r'\?sort=',
        # 2. 匹配 Apache/Nginx 默认目录列表的排序参数 (例如 ?C=N;O=D, ?C=M;O=A)
        r'\?C=[A-Z];O=[A-Z]', 
        # 3. 过滤无用的文件后缀（如果是RAG，通常只需要网页和PDF，不需要代码、压缩包和日志）
        r'\.(zip|tar|gz|rar|bin|log|bak|iso|exe|csv|sql|txt)$', 
        # 4. 可选：如果确定 archive（历史存档）不需要，可以取消下面这行的注释
        # r'/archive/', 
    ]
    
    # 编译正则表达式以提高匹配速度
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in unwanted_patterns]

    total_files = 0
    rejected_files = 0
    error_files = 0

    print("开始进行 URL 清洗...")

    # 遍历目录中的所有 json 文件
    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        total_files += 1
        filepath = os.path.join(input_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            url = data.get("url", "")
            
            # 如果没有 url 字段，或者 url 命中了黑名单规则，则判定为垃圾数据
            is_bad_url = False
            if not url:
                is_bad_url = True
            else:
                for pattern in compiled_patterns:
                    if pattern.search(url):
                        is_bad_url = True
                        break
            
            # 如果命中规则，移动到 rejected 文件夹
            if is_bad_url:
                rejected_files += 1
                shutil.move(filepath, os.path.join(rejected_dir, filename))
                # print(f"已过滤: {url}") # 如果你想看具体过滤了哪些，可以取消注释
                
        except json.JSONDecodeError:
            print(f"JSON解析错误 (文件已损坏): {filename}")
            error_files += 1
        except Exception as e:
            print(f"处理文件出错 {filename}: {e}")
            error_files += 1

        # 每处理 1000 个文件打印一次进度
        if total_files % 1000 == 0:
            print(f"已处理 {total_files} 个文件...")

    # 打印最终统计结果
    kept_files = total_files - rejected_files - error_files
    print("\n" + "="*30)
    print("URL 清洗完成！统计信息如下：")
    print(f"总处理文件数: {total_files}")
    print(f"保留文件数:   {kept_files} ({(kept_files/total_files)*100:.2f}%)")
    print(f"剔除文件数:   {rejected_files} ({(rejected_files/total_files)*100:.2f}%) -> 存放在 {rejected_dir}")
    if error_files > 0:
        print(f"错误文件数:   {error_files}")
    print("="*30)

if __name__ == "__main__":
    clean_urls()