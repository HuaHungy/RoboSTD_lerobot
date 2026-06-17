#!/bin/bash

# 检查是否提供了目录参数，如果没有则使用当前目录
TARGET_DIR="${1:-.}"
UNDO_FILE="undo_rename.sh"

# 初始化还原脚本
echo "#!/bin/bash" > "$UNDO_FILE"
echo "echo '开始还原...'" >> "$UNDO_FILE"
chmod +x "$UNDO_FILE"

echo "开始在 $TARGET_DIR 中查找并重命名包含 '1732' 的文件夹..."

# 使用 find 命令查找包含 1732 的文件夹，-depth 确保先处理子目录
find "$TARGET_DIR" -depth -type d -name "*1732*" | while read -r dir; do
    # 获取父目录和旧文件夹名
    parent_dir=$(dirname "$dir")
    old_name=$(basename "$dir")
    
    # 将 1732 替换为 1731
    new_name=${old_name//1732/1731}
    new_dir="$parent_dir/$new_name"
    
    # 执行重命名
    mv "$dir" "$new_dir"
    echo "已重命名: $old_name -> $new_name"
    
    # 将还原命令写入 undo 脚本
    echo "mv \"$new_dir\" \"$dir\"" >> "$UNDO_FILE"
    echo "echo '已还原: $new_name -> $old_name'" >> "$UNDO_FILE"
done

echo "重命名完成！已自动生成还原脚本: $UNDO_FILE"
echo "如果需要撤销操作，请运行: ./$UNDO_FILE"
