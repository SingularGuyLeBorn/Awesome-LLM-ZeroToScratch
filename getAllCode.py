import os


def merge_files_to_txt(folder_path, target_extensions, output_filename, excluded_folders=None):
    """
    将指定文件夹（包括子文件夹）下所有指定后缀文件的内容合并到一个 TXT 文件中。

    参数:
    folder_path (str): 要遍历的文件夹路径。
    target_extensions (list): 要查找的文件的后缀名列表（例如：['.txt', '.py', '.md']）。
    output_filename (str): 输出的 TXT 文件的名称。
    excluded_folders (list, optional): 要跳过的文件夹名称列表。默认为 None (不跳过任何文件夹)。
    """
    if excluded_folders is None:
        excluded_folders = []  # 如果没有提供，则初始化为空列表

    try:
        # 确保每个目标后缀都以 '.' 开头
        # 使用列表推导式更简洁地处理
        processed_extensions = [ext if ext.startswith('.') else '.' + ext for ext in target_extensions]

        # 获取输出文件的完整路径
        output_filepath = os.path.join(folder_path, output_filename)

        # 记录实际跳过的文件夹
        actual_excluded_dirs = []

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # os.walk() 会遍历指定目录下的所有文件夹和文件
            # dirnames 是当前目录下的子目录列表
            for dirpath, dirnames, filenames in os.walk(folder_path):
                # ==========================================================
                # 关键修改：根据 excluded_folders 参数排除文件夹
                # 需要在这里创建一个 dirnames 的副本，因为直接修改 dirnames 列表会影响 os.walk 的遍历行为
                # 或者更直接的方式是修改 dirnames 自身，让 os.walk 不再进入这些目录
                # 我们直接从 dirnames 中移除，以跳过这些目录

                # 创建一个需要移除的目录列表，避免在遍历的同时修改列表
                dirs_to_remove = [d for d in dirnames if d in excluded_folders]
                for d in dirs_to_remove:
                    dirnames.remove(d)
                    actual_excluded_dirs.append(os.path.join(dirpath, d))  # 记录完整路径

                # ==========================================================

                for filename in filenames:
                    # 检查文件是否以任一目标后缀结尾
                    # 同时确保空后缀（无后缀文件）也能被正确处理
                    is_match = False
                    if '' in processed_extensions and '.' not in filename:  # 如果包含空后缀且文件无后缀
                        is_match = True
                    elif any(filename.endswith(ext) for ext in processed_extensions):  # 如果有匹配的后缀
                        is_match = True

                    if is_match:
                        filepath = os.path.join(dirpath, filename)

                        # 写入文件分隔符和文件名，方便区分不同文件的内容
                        outfile.write(f'{"=" * 20}\n')
                        outfile.write(f'文件: {filepath}\n')
                        outfile.write(f'{"=" * 20}\n\n')

                        try:
                            with open(filepath, 'r', encoding='utf-8') as infile:
                                outfile.write(infile.read())
                                outfile.write('\n\n')  # 在每个文件内容后添加空行，增加可读性
                        except Exception as e:
                            outfile.write(f'读取文件 {filepath} 时出错: {e}\n\n')

        print(f'成功！所有 {", ".join(processed_extensions)} 文件的内容已合并到 {output_filepath}')
        if actual_excluded_dirs:
            print(f'注意：已跳过以下文件夹及其内容:')
            for d in set(actual_excluded_dirs):  # 使用set去重，避免打印重复路径
                print(f'- {d}')
        else:
            print(f'没有指定或实际跳过任何文件夹。')

    except FileNotFoundError:
        print(f'错误：文件夹 "{folder_path}" 不存在。')
    except Exception as e:
        print(f'发生未知错误: {e}')


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 设置要遍历的文件夹路径
    source_folder = '.'  # '.' 表示当前脚本所在的文件夹

    # 2. 设置要查找的文件后缀名列表
    file_extensions = ['.py', 'sh', 'yaml', 'yml']

    # 3. 设置输出的 TXT 文件名
    output_file = 'merged_multiple_contents_custom_exclude.txt'

    # 4. 设置要跳过的文件夹名称列表
    #    这里可以添加任何你不想遍历的文件夹名称
    folders_to_exclude = ['.venv', '__pycache__', 'node_modules', '.git']

    # 5. 调用函数
    merge_files_to_txt(source_folder, file_extensions, output_file, folders_to_exclude)