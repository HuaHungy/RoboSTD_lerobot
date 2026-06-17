# # 左臂can0通信连接
# # 1. 先关闭接口
# sudo ip link set can0 down
# # 3. 激活并设置波特率为 1M
# sudo ip link set can0 up type can bitrate 1000000

# # 右臂can1通信连接
# # 1. 先关闭接口
# sudo ip link set can1 down
# # 3. 激活并设置波特率为 1M
# sudo ip link set can1 up type can bitrate 1000000

# 5.8修改：

sudo ifconfig can1 down
sudo ifconfig can_right down

sudo ip link set can_right name can0

# 配置 can0（左臂），比特率根据实际需要调整
sudo ip link set can0 type can bitrate 1000000
sudo ifconfig can0 up

# 配置 can1（右臂）
sudo ip link set can1 type can bitrate 1000000
sudo ifconfig can1 up
