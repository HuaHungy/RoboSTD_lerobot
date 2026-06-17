# 模拟服务端来发送相关指令控制机械臂运动
cd src
python mock_server.py


# 新建一个终端窗口
# 模拟客户端接受消息，可观测机械臂小范围抖动
cd src
python test_bi_piper_client.py