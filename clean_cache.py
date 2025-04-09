import os
import shutil
import glob


def clean_python_cache():
    # 要清理的目录列表
    directories = ["llumnix", "tests", "downlaod"]

    # 要删除的文件模式
    patterns = ["*.pyc", "*.pyo", "*.pyd", "*.so", "__pycache__"]

    for directory in directories:
        if not os.path.exists(directory):
            continue

        print(f"Cleaning directory: {directory}")

        # 删除 __pycache__ 目录
        for root, dirs, files in os.walk(directory):
            if "__pycache__" in dirs:
                cache_dir = os.path.join(root, "__pycache__")
                print(f"Removing: {cache_dir}")
                shutil.rmtree(cache_dir)

        # 删除其他缓存文件
        for pattern in patterns:
            for file in glob.glob(
                os.path.join(directory, "**", pattern), recursive=True
            ):
                if os.path.isdir(file):
                    print(f"Removing directory: {file}")
                    shutil.rmtree(file)
                else:
                    print(f"Removing file: {file}")
                    os.remove(file)


if __name__ == "__main__":
    clean_python_cache()
    print("Python cache cleanup completed!")
