import cv2
import platform
import glob
import sys
import os
import argparse

def scan_cameras():
    print("正在扫描可用的相机设备，请稍候...")
    indices = []
    if platform.system() == "Linux":
        paths = glob.glob('/dev/video*')
        for p in paths:
            try:
                idx = int(p.replace('/dev/video', ''))
                indices.append(idx)
            except ValueError:
                pass
        indices.sort()
    else:
        indices = list(range(20))

    available = []
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            # 尝试读取一帧来确认是不是真实的视频捕获设备
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available

def capture_interactive(cameras):
    print("=" * 50)
    print("操作说明：")
    print("  [ n ] 或 [ 空格键 ] : 切换到【下一个相机】")
    print("  [ q ] 或 [ Esc键 ]  : 退出脚本")
    print("=" * 50)

    current_pos = 0
    while True:
        idx = cameras[current_pos]
        print(f"\n---> 正在打开相机 Index: {idx} (对应设备路径: /dev/video{idx})")
        
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"无法打开相机 {idx}，自动跳过...")
            current_pos = (current_pos + 1) % len(cameras)
            continue

        # 设置预览画面的默认分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = f"Camera Viewer - Index {idx}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        quit_all = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"无法从相机 {idx} 读取画面。")
                break

            # 在画面上添加提示文本
            cv2.putText(frame, f"Index: {idx}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Next: 'n' | Quit: 'q'", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)
            
            # 等待键盘输入
            key = cv2.waitKey(30) & 0xFF
            
            if key in [ord('q'), 27]:  # 'q' 或 ESC 键
                quit_all = True
                break
            elif key in [ord('n'), 32]:  # 'n' 或 空格键
                break

        cap.release()
        cv2.destroyAllWindows()

        if quit_all:
            print("\n已退出交互式相机检测脚本。")
            break

        # 切换到下一个相机
        current_pos = (current_pos + 1) % len(cameras)

def capture_images(cameras):
    print("由于处于服务器模式，将把相机画面截取为图片保存在 scripts 目录下。")
    print("=" * 50)
    
    for idx in cameras:
        print(f"\n---> 正在打开相机 Index: {idx} (对应设备路径: /dev/video{idx})")
        
        # 尝试指定不同的后端 (V4L2 可能会对有些相机兼容性更好)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            # 失败则回退默认
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"无法打开相机 {idx}，自动跳过...")
                continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 多读取几帧以让相机曝光和白平衡稳定
        for _ in range(5):
            ret, frame = cap.read()

        if not ret:
            print(f"无法从相机 {idx} 读取画面。")
            cap.release()
            continue

        # 在画面上添加提示文本
        cv2.putText(frame, f"Camera Index: {idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        save_path = os.path.join(os.path.dirname(__file__), f"camera_index_{idx}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"已保存当前相机截图: {save_path}")

        cap.release()

    print("\n所有可用相机截图已保存完毕，请到 scripts 目录下查看图片确认每个 Index 对应的镜头位置！")

def main():
    parser = argparse.ArgumentParser(description="交互式相机检测脚本")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["local", "server"], 
        default="local",
        help="运行模式: 'local' 会弹出实时视频窗口(需GUI环境); 'server' 会直接截图保存"
    )
    args = parser.parse_args()

    cameras = scan_cameras()
    if not cameras:
        print("未检测到任何可用的相机设备！")
        sys.exit(1)

    print(f"\n成功检测到 {len(cameras)} 个相机，对应的 Index 编号为: {cameras}")
    
    if args.mode == "local":
        capture_interactive(cameras)
    else:
        capture_images(cameras)

if __name__ == "__main__":
    main()
