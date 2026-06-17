#!/bin/bash

TARGET_DIR="/home/agilex/DoRobot/dataset/20260414/user/Agilex_Cobot_Magic_pick_up_bottle_2_1731"

echo "开始在 $TARGET_DIR 中恢复误命名的文件夹..."

cd "$TARGET_DIR" || exit 1

# 遍历 484366 到 484405 的文件夹
for i in {484366..484405}; do
    old_dir="Agilex_Cobot_Magic_pick_up_bottle_2_1731_$i"
    new_dir="Agilex_Cobot_Magic_pick_up_bottle_2_1732_$i"
    
    # 检查重命名后的文件夹（现在叫 1731_xxx）是否存在
    if [ -d "$old_dir" ]; then
        mv "$old_dir" "$new_dir"
        echo "已恢复: $old_dir -> $new_dir"
    fi
done

echo "恢复完成！"
