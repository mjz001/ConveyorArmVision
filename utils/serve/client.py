from fastapi import FastAPI,HTTPException
import base64
import cv2
import numpy as np
import argparse
import aiohttp
import time
import io
import asyncio

def encodebase64(frame):

    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64_str = base64.b64encode(buffer).decode('utf-8')
        return image_base64_str
    except Exception as e:
        raise Exception(f"Error encoding image to base64: {str(e)}")

def draw_tracked_boxes(frame, tracked_bboxes, contour_list, class_names):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for i, box in enumerate(tracked_bboxes):
            x1, y1, x2, y2, track_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 生成随机颜色，但强制禁止红色 (BGR格式)
            np.random.seed(int(track_id))
            b = np.random.randint(100, 255)
            g = np.random.randint(100, 255)
            r = np.random.randint(0, 100)  # 红色通道调低，绝对不会出现纯红
            color = (b, g, r)

            # 获取文字 - 确保不会超出class_names的范围
            cls_text = class_names[i] if i < len(class_names) else "unknown"
            id_text = f"ID:{int(track_id)}"
            info_text = f"{id_text} | {cls_text}"

            # 计算文字背景大小
            (text_w, text_h), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
            text_bg_x1 = x1
            text_bg_y1 = y1 - text_h - baseline - 2
            text_bg_x2 = x1 + text_w
            text_bg_y2 = y1

            # 绘制背景
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)

            # 白色文字
            cv2.putText(frame, info_text,
                        (x1, text_bg_y2 - baseline),
                        font, font_scale, (255,255,255), thickness)

            # 绘制目标框（非红色）
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 轮廓 固定 红色 (0,0,255)
            if i < len(contour_list):
                cnt = contour_list[i].astype(np.int32) if isinstance(contour_list[i], np.ndarray) else np.array(contour_list[i]).astype(np.int32)
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)

        return frame
    
def cv_imshow(name,frame):
    cv2.imshow(name,frame)
    if cv2.waitKey(1)==27:  # 按ESC键退出
        return False
    return True

class APIClient:
    def __init__(self, base_url, timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def track_single_image(self, base64_image_data):
        """处理单张图像的追踪请求"""
        url = f"{self.base_url}/track"
        payload = {"image": base64_image_data}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "data": result}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timed out"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"Client error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    async def track_multiple_images(self, base64_image_data):
        """对图像进行标注并返回带标注的图像"""
        url = f"{self.base_url}/track_multiple"
        payload = {"image": base64_image_data}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "data": result}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timed out"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"Client error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    async def health_check(self):
        """健康检查请求"""
        url = f"{self.base_url}/health"
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "data": result}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Health check timed out"}
        except aiohttp.ClientError as e:
            return {"success": False, "error": f"Health check client error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Health check unexpected error: {str(e)}"}

async def send_request(args):
    #参数配置区域
    client = APIClient(args.url, timeout=args.timeout)
    video_path = args.video_path  # 填0为摄像头，也可以填视频路径   
    cap = cv2.VideoCapture(video_path)
    
    #检查健康状态
    health_result = await client.health_check()
    if not health_result["success"]:
        print(f"Health check failed: {health_result['error']}")
        return
    print("Service health check passed.")
    
    #创建异步会话
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        prev_time = time.time()
        frame_count =0 
        fps = 0
        try:
            while True:
                ret,frame = cap.read()
                if not ret or frame is None:
                    print("视频读取完毕或失败")
                    break
                base64_str = encodebase64(frame)
                try:
                    # 根据路由参数选择不同的处理方法
                    if args.route == "/track":
                        result = await client.track_single_image(base64_str)
                    elif args.route == "/track_multiple":
                        result = await client.track_multiple_images(base64_str)
                    else:
                        print(f"Unknown route: {args.route}")
                        break
                    
                    if result["success"]:
                        response_data = result["data"]
                        print(f"Received response for {args.route}: Success")
                        
                        # 根据不同路由处理返回数据
                        if args.route == "/track":
                            tracker_boxes = response_data["tracker_boxes"]
                            contour_list = response_data["contour_list"]
                            class_names = response_data["class_names"]
                            image = draw_tracked_boxes(frame, tracker_boxes, contour_list, class_names)
                            
                        elif args.route == "/track_multiple":
                            # 解码返回的base64图像
                            annotated_base64 = response_data["annotated_image"]
                            image_bytes = base64.b64decode(annotated_base64)
                            np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
                            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        #帧率计算
                        current_time = time.time()
                        frame_count += 1
                        if frame_count % 10 == 0:
                            dt = current_time - prev_time
                            fps = 10.0 / dt if dt > 0 else 0
                            prev_time = current_time  
                        cv2.putText(image, f"FPS: {fps:.1f}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if not cv_imshow("Tracking", image):  # 显示
                            break
                            
                    else:
                        print(f"Error processing request: {result['error']}")
                        
                except aiohttp.ClientError as e:
                    print(f"Error sending request: {e}")
                except Exception as e:
                    print(f"Unexpected error during request: {e}")
                    
        except Exception as e:
            print(f"Error processing frame: {e}")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

        
               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='conveyor.mp4', help='path to input video')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000', help='base URL of the API service')
    parser.add_argument('--route', type=str, default='/track', choices=['/health','/track', '/track_multiple'], help='API route to use')
    parser.add_argument('--timeout', type=int, default=30, help='request timeout in seconds')
    args = parser.parse_args()
    
    asyncio.run(send_request(args))