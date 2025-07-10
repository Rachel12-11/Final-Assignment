import os
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog, scrolledtext
import tempfile
import threading
import json
import re
import time
from datetime import timedelta
from zhconv import convert
import whisper
import numpy as np
from sklearn.cluster import KMeans
import librosa

class SubtitleGenerator:
    """字幕生成核心功能类"""
    
    def __init__(self, ffmpeg_path="ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self.whisper_model = None
        self.model_size = "base"
        self.speaker_embeddings = {}
        self.speaker_threshold = 0.9
        self.min_speaker_samples = 1
    
    def set_ffmpeg_path(self, path):
        """设置FFmpeg路径"""
        self.ffmpeg_path = path
        if self.ffmpeg_path and os.path.exists(self.ffmpeg_path):
            ffmpeg_dir = os.path.dirname(self.ffmpeg_path)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
            print(f"已为Whisper设置FFmpeg路径: {self.ffmpeg_path}")
        else:
            print(f"警告: 指定的FFmpeg路径不存在: {self.ffmpeg_path}")
    
    def check_ffmpeg(self):
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True
            )
            print(f"FFmpeg版本: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"检查FFmpeg失败: {e.stderr}")
            return False
        except Exception as e:
            print(f"执行FFmpeg检查时出错: {str(e)}")
            return False
    
    def extract_audio(self, video_path, audio_output_path):
        """从视频中提取音频"""
        print(f"开始提取音频: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在 - {video_path}")
            return False
        
        output_dir = os.path.dirname(audio_output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                audio_output_path
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )
            print(f"音频提取成功: {audio_output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"音频提取失败: {e.stderr}")
            return False
        except Exception as e:
            print(f"执行FFmpeg时发生未知错误: {str(e)}")
            return False
    
    def load_whisper_model(self, model_size="base"):
        """加载Whisper模型"""
        if self.whisper_model and self.model_size == model_size:
            return True
            
        try:
            print(f"加载Whisper模型: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
            self.model_size = model_size
            return True
        except Exception as e:
            print(f"加载Whisper模型失败: {str(e)}")
            self.whisper_model = None
            return False
    
    def transcribe_audio(self, audio_path, language="zh"):
        """使用Whisper模型进行语音识别"""
        print(f"开始语音识别: {audio_path}")
        
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件不存在 - {audio_path}")
            return []
            
        if not self.whisper_model:
            if not self.load_whisper_model(self.model_size):
                return []
                
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                temperature=0.1,
                best_of=5,
                beam_size=5
            )
            
            subtitles = []
            for segment in result["segments"]:
                simplified_text = convert(segment["text"], 'zh-cn')
                processed_text = self._post_process_text(simplified_text)
                
                subtitles.append({
                    "text": processed_text,
                    "start": segment["start"],
                    "end": segment["end"]
                })
                
            print(f"语音识别完成，生成 {len(subtitles)} 条字幕")
            return subtitles
            
        except Exception as e:
            print(f"语音识别异常: {str(e)}")
            return []
    
    def _post_process_text(self, text):
        """文本后处理，优化字幕质量"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'，，', '，', text)
        text = re.sub(r'。。', '。', text)
        text = re.sub(r'！！', '！', text)
        text = re.sub(r'？？', '？', text)
        text = re.sub(r'([，。！？：；,.?!:;])([^\s])', r'\1 \2', text)
        return text
    
    def extract_audio_features(self, audio_path, start_time, end_time):
        """提取指定时间段的音频特征用于说话人识别"""
        try:
            # 使用librosa加载指定时间段的音频
            y, sr = librosa.load(
                audio_path,
                sr=16000,
                offset=start_time,
                duration=end_time - start_time
            )
            # 提取MFCC特征（梅尔频率倒谱系数）
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # 计算特征均值作为嵌入向量
            embedding = np.mean(mfcc, axis=1)
            # 归一化处理
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"提取音频特征失败: {str(e)}")
            return None
    
    def classify_speaker(self, embedding):
        """根据音频特征分类说话人"""
        if not self.speaker_embeddings or len(self.speaker_embeddings) < self.min_speaker_samples:
            return None  # 样本不足时不分类
            
        # 计算与已知说话人的相似度
        similarities = {}
        for speaker, embeddings in self.speaker_embeddings.items():
            # 计算该说话人的平均特征
            mean_embedding = np.mean(embeddings, axis=0)
            # 余弦相似度
            similarity = np.dot(embedding, mean_embedding)
            similarities[speaker] = similarity
            
        # 找到最相似的说话人
        best_speaker = max(similarities.items(), key=lambda x: x[1])
        if best_speaker[1] > self.speaker_threshold:
            return best_speaker[0]
        return None  # 低于阈值则视为新说话人
    
    def update_speaker_embeddings(self, speaker, embedding):
        """更新说话人特征库"""
        if speaker not in self.speaker_embeddings:
            self.speaker_embeddings[speaker] = []
        self.speaker_embeddings[speaker].append(embedding)
        # 限制每个说话人的样本数量（只保留最新20个）
        if len(self.speaker_embeddings[speaker]) > 20:
            self.speaker_embeddings[speaker] = self.speaker_embeddings[speaker][-20:]
    
    def auto_detect_speakers(self, subtitles, audio_path, n_speakers=2):
        """自动识别所有字幕的说话人"""
        if not subtitles or not audio_path or not os.path.exists(audio_path):
            return subtitles
            
        total_segments = len(subtitles)
        if total_segments == 0:
            return subtitles
            
        print(f"开始分析 {total_segments} 个语音片段，预设说话人数量: {n_speakers}")
        
        # 1. 提取所有片段的音频特征
        valid_subtitles = []
        embeddings = []
        
        for i, subtitle in enumerate(subtitles):
            embedding = self.extract_audio_features(audio_path, subtitle['start'], subtitle['end'])
            if embedding is not None:
                valid_subtitles.append(subtitle)
                embeddings.append(embedding)
                
        if not embeddings:
            print("警告: 未提取到有效的音频特征")
            return subtitles
            
        embeddings = np.array(embeddings)
        n_speakers = min(max(1, n_speakers), len(embeddings))
        
        # 2. KMeans聚类
        print(f"正在聚类为{n_speakers}个说话人...")
        kmeans = KMeans(n_clusters=n_speakers, random_state=0, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # 3. 生成说话人标签
        speaker_names = [f"说话人{i+1}" for i in range(n_speakers)]
        
        # 4. 分配说话人到字幕
        for i, (subtitle, label) in enumerate(zip(valid_subtitles, labels)):
            subtitle['speaker'] = speaker_names[label]
            
        # 5. 更新特征库
        self.speaker_embeddings = {name: [] for name in speaker_names}
        for i, (subtitle, label) in enumerate(zip(valid_subtitles, labels)):
            self.speaker_embeddings[speaker_names[label]].append(embeddings[i])
            
        print(f"说话人识别完成，共{n_speakers}个说话人")
        return subtitles
    
    def generate_srt(self, subtitles, srt_path):
        """生成带说话人信息的SRT格式字幕文件"""
        print(f"开始生成SRT文件: {srt_path}")
        
        output_dir = os.path.dirname(srt_path)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles, 1):
                    start_time = self.format_time(subtitle['start'])
                    end_time = self.format_time(subtitle['end'])
                    text = subtitle['text']
                    
                    if 'speaker' in subtitle and subtitle['speaker']:
                        text = f"{subtitle['speaker']}: {text}"
                        
                    f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
            print(f"SRT文件生成成功")
            return True
        except Exception as e:
            print(f"生成字幕文件失败: {str(e)}")
            return False
    
    def merge_subtitles(self, video_path, srt_path, output_path, font_name="SimHei", font_size=24, position="底部"):
        """将字幕合并到视频中"""
        print(f"开始合并字幕: {srt_path} 到 {video_path}")
        
        for file_path, file_type in [(video_path, "视频"), (srt_path, "字幕")]:
            if not os.path.exists(file_path):
                print(f"错误: {file_type}文件不存在 - {file_path}")
                return False
                
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            video_info = self._get_video_info(video_path)
            if not video_info:
                print("警告: 无法获取视频尺寸信息，使用默认值")
                width, height = 1920, 1080
            else:
                width, height = video_info['width'], video_info['height']
                print(f"视频尺寸: {width}x{height}")
                
            position_map = {
                "顶部": "y=100",
                "中间": "y=(h-text_h)/2",
                "底部": "y=h-text_h-50"
            }
            position_setting = position_map.get(position, "y=h-text_h-50")
            
            escaped_srt_path = srt_path.replace('\\', '\\\\').replace(':', '\\:').replace('[', '\\[').replace(']', '\\]')
            
            srt_filter = (
                f"subtitles='{escaped_srt_path}':"
                f"force_style='FontName={font_name},"
                f"FontSize={font_size},"
                f"PrimaryColour=&HFFFFFF&,"
                f"BackColour=&H80000000&,"
                f"Outline=1,"
                f"Alignment=2,"
                f"{position_setting}'"
            )
            
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vf', srt_filter,
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600
            )
            print(f"字幕合并成功: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"字幕合并失败: {e.stderr}")
            return False
        except Exception as e:
            print(f"执行FFmpeg时发生未知错误: {str(e)}")
            return False
    
    def _get_video_info(self, video_path):
        """获取视频信息，主要是尺寸"""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            info = json.loads(result.stdout)
            if 'streams' in info and len(info['streams']) > 0:
                stream = info['streams'][0]
                return {
                    'width': stream.get('width', 1920),
                    'height': stream.get('height', 1080)
                }
            return None
        except Exception as e:
            print(f"获取视频信息失败: {str(e)}")
            return None
    
    def format_time(self, seconds):
        """将秒转换为SRT格式的时间字符串"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def save_subtitles_json(self, subtitles, json_path):
        """保存带说话人信息的字幕数据为JSON格式"""
        print(f"开始保存JSON数据: {json_path}")
        
        output_dir = os.path.dirname(json_path)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(subtitles, f, ensure_ascii=False, indent=2)
            print(f"JSON数据保存成功")
            return True
        except Exception as e:
            print(f"保存JSON数据失败: {str(e)}")
            return False

class SubtitleGUI:
    """字幕生成器图形界面类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("智能字幕生成器（带说话人标记）")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")
        
        # 创建主画布和滚动条
        self.canvas = tk.Canvas(self.root, bg="#f0f0f0", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # 创建画布上的框架
        self.main_frame = ttk.Frame(self.canvas, padding=10)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        
        # 设置中文字体
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TProgressbar", thickness=20)
        
        # 初始化变量
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.srt_path = tk.StringVar()
        self.json_path = tk.StringVar()
        self.video_selected = False
        self.subtitles_generated = False
        self.processing = False
        self.speaker_mode = tk.BooleanVar(value=False)
        self.speakers = ["旁白"]
        self.current_speaker = tk.StringVar(value=self.speakers[0])
        self.subtitles_with_speakers = []
        
        # 字幕生成器核心实例
        self.generator = SubtitleGenerator()
        
        # Whisper模型配置
        self.whisper_model_size = "base"
        
        # 创建界面
        self.create_widgets()
        
        # 绑定事件
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
    
    def on_frame_configure(self, event):
        """当框架大小变化时，更新画布的滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """当画布大小变化时，调整框架宽度"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width - 20)
    
    def on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        if event.state & 0x1:  # 按下Shift键
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def create_widgets(self):
        # 标题
        title_frame = ttk.Frame(self.main_frame, padding=(20, 10))
        title_frame.pack(fill="x")
        
        ttk.Label(title_frame, text="智能字幕生成器（带说话人标记）", font=("SimHei", 16, "bold")).pack()
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(self.main_frame, text="文件选择", padding=10)
        file_frame.pack(fill="x", padx=20, pady=5)
        
        # 输入视频
        ttk.Label(file_frame, text="输入视频:").grid(row=0, column=0, sticky="w", pady=2)
        input_frame = ttk.Frame(file_frame)
        input_frame.grid(row=0, column=1, sticky="ew", pady=2)
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=35, state="readonly").pack(side="left", fill="x", expand=True)
        ttk.Button(input_frame, text="浏览...", command=self.browse_input).pack(side="left", padx=5)
        
        # FFmpeg路径设置
        ttk.Label(file_frame, text="FFmpeg路径:").grid(row=1, column=0, sticky="w", pady=2)
        ffmpeg_frame = ttk.Frame(file_frame)
        ffmpeg_frame.grid(row=1, column=1, sticky="ew", pady=2)
        
        self.ffmpeg_var = tk.StringVar(value=self.generator.ffmpeg_path)
        ttk.Entry(ffmpeg_frame, textvariable=self.ffmpeg_var, width=35).pack(side="left", fill="x", expand=True)
        ttk.Button(ffmpeg_frame, text="浏览...", command=self.browse_ffmpeg).pack(side="left", padx=5)
        
        # 确定按钮
        confirm_frame = ttk.Frame(file_frame)
        confirm_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        self.confirm_button = ttk.Button(confirm_frame, text="确定", command=self.confirm_selection, state="disabled")
        self.confirm_button.pack()
        
        # 设置列权重
        file_frame.columnconfigure(1, weight=1)
        
        # 输出设置
        self.output_info_frame = ttk.LabelFrame(self.main_frame, text="输出设置", padding=10)
        
        # 输出视频
        ttk.Label(self.output_info_frame, text="输出视频:").grid(row=0, column=0, sticky="w", pady=2)
        output_frame = ttk.Frame(self.output_info_frame)
        output_frame.grid(row=0, column=1, sticky="ew", pady=2)
        
        ttk.Entry(output_frame, textvariable=self.output_path, width=40).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="浏览...", command=self.browse_output).pack(side="left", padx=5)
        
        # SRT字幕文件
        ttk.Label(self.output_info_frame, text="SRT字幕:").grid(row=1, column=0, sticky="w", pady=2)
        srt_frame = ttk.Frame(self.output_info_frame)
        srt_frame.grid(row=1, column=1, sticky="ew", pady=2)
        
        ttk.Entry(srt_frame, textvariable=self.srt_path, width=40).pack(side="left", fill="x", expand=True)
        ttk.Button(srt_frame, text="浏览...", command=self.browse_srt).pack(side="left", padx=5)
        
        # JSON字幕数据
        ttk.Label(self.output_info_frame, text="JSON数据:").grid(row=2, column=0, sticky="w", pady=2)
        json_frame = ttk.Frame(self.output_info_frame)
        json_frame.grid(row=2, column=1, sticky="ew", pady=2)
        
        ttk.Entry(json_frame, textvariable=self.json_path, width=40).pack(side="left", fill="x", expand=True)
        ttk.Button(json_frame, text="浏览...", command=self.browse_json).pack(side="left", padx=5)
        
        # 设置列权重
        self.output_info_frame.columnconfigure(1, weight=1)
        
        # 选项区域
        options_frame = ttk.LabelFrame(self.main_frame, text="选项", padding=10)
        options_frame.pack(fill="x", padx=20, pady=5)
        
        # Whisper模型选择
        model_frame = ttk.Frame(options_frame)
        model_frame.pack(fill="x", pady=2)
        
        ttk.Label(model_frame, text="Whisper模型:").pack(side="left")
        
        self.model_var = tk.StringVar(value=self.whisper_model_size)
        model_sizes = ["tiny", "base", "small", "medium", "large"]
        for size in model_sizes:
            ttk.Radiobutton(
                model_frame,
                text=size,
                variable=self.model_var,
                value=size
            ).pack(side="left", padx=5)
        
        # 说话人模式
        speaker_frame = ttk.Frame(options_frame)
        speaker_frame.pack(fill="x", pady=2)
        
        ttk.Checkbutton(speaker_frame, text="启用说话人标记", variable=self.speaker_mode,
                       command=self.toggle_speaker_mode).pack(side="left")
        
        ttk.Button(speaker_frame, text="管理说话人", command=self.manage_speakers).pack(side="left", padx=5)
        ttk.Button(speaker_frame, text="自动识别说话人", command=self.auto_detect_speakers).pack(side="left", padx=5)
        ttk.Button(speaker_frame, text="调整说话人数量", command=self.adjust_speaker_count).pack(side="left", padx=5)
        
        # 预设说话人数量
        ttk.Label(speaker_frame, text="预设说话人数量:").pack(side="left", padx=(15, 0))
        self.preset_speaker_count = tk.IntVar(value=2)
        preset_spin = ttk.Spinbox(speaker_frame, from_=1, to=10, width=5, textvariable=self.preset_speaker_count)
        preset_spin.pack(side="left", padx=2)
        
        # 说话人相似度阈值设置
        ttk.Label(speaker_frame, text="相似度阈值:").pack(side="left", padx=(15, 0))
        self.speaker_threshold_var = tk.DoubleVar(value=self.generator.speaker_threshold)
        threshold_spin = ttk.Spinbox(speaker_frame, from_=0.5, to=0.99, increment=0.01, width=5, textvariable=self.speaker_threshold_var)
        threshold_spin.pack(side="left", padx=2)
        def update_threshold(*args):
            self.generator.speaker_threshold = self.speaker_threshold_var.get()
        self.speaker_threshold_var.trace_add("write", update_threshold)
        
        # 说话人选择（初始隐藏）
        self.speaker_selector_frame = ttk.Frame(options_frame)
        
        ttk.Label(self.speaker_selector_frame, text="当前说话人:").pack(side="left")
        
        self.speaker_combo = ttk.Combobox(self.speaker_selector_frame, textvariable=self.current_speaker,
                                          values=self.speakers, state="readonly", width=15)
        self.speaker_combo.pack(side="left", padx=5)
        
        ttk.Button(self.speaker_selector_frame, text="应用到选中字幕",
                  command=self.apply_speaker_to_selection).pack(side="left", padx=5)
        
        ttk.Button(self.speaker_selector_frame, text="应用到所有字幕",
                  command=self.apply_speaker_to_all).pack(side="left", padx=5)
        
        # 字幕样式设置
        style_frame = ttk.Frame(options_frame)
        style_frame.pack(fill="x", pady=2)
        
        ttk.Label(style_frame, text="字幕样式:").pack(side="left")
        
        self.font_var = tk.StringVar(value="SimHei")
        ttk.Entry(style_frame, textvariable=self.font_var, width=10).pack(side="left", padx=5)
        ttk.Label(style_frame, text="字体大小:").pack(side="left", padx=(5, 0))
        
        self.fontsize_var = tk.IntVar(value=24)
        ttk.Spinbox(style_frame, from_=10, to=40, width=5, textvariable=self.fontsize_var).pack(side="left", padx=5)
        
        ttk.Label(style_frame, text="位置:").pack(side="left", padx=(10, 0))
        
        self.position_var = tk.StringVar(value="底部")
        positions = ["顶部", "中间", "底部"]
        for pos in positions:
            ttk.Radiobutton(
                style_frame,
                text=pos,
                variable=self.position_var,
                value=pos
            ).pack(side="left", padx=5)
        
        # 按钮区域
        button_frame = ttk.Frame(self.main_frame, padding=10)
        button_frame.pack(fill="x", padx=20, pady=5)
        
        self.process_subtitle_button = ttk.Button(button_frame, text="生成字幕", command=self.process_subtitles,
                                                  state="disabled")
        self.process_subtitle_button.pack(side="left", padx=5)
        
        self.merge_video_button = ttk.Button(button_frame, text="合并字幕到视频", command=self.merge_video,
                                             state="disabled")
        self.merge_video_button.pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="保存修改", command=self.save_subtitle_changes).pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side="right", padx=5)
        
        # 进度条
        progress_frame = ttk.Frame(self.main_frame, padding=10)
        progress_frame.pack(fill="x", padx=20, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100)
        self.progress_bar.pack(fill="x")
        
        self.status_var = tk.StringVar(value="请选择视频文件")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w", pady=2)
        
        # 字幕编辑区域
        subtitle_frame = ttk.LabelFrame(self.main_frame, text="字幕编辑（可直接修改文本）", padding=10)
        subtitle_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        # 可滚动的Text组件
        self.subtitle_text = scrolledtext.ScrolledText(subtitle_frame, height=15, width=90, wrap=tk.WORD)
        self.subtitle_text.pack(fill="both", expand=True)
        
        # 添加右键菜单
        self.subtitle_text.bind("<Button-3>", self.show_context_menu)
        self.context_menu = tk.Menu(self.subtitle_text, tearoff=0)
        self.context_menu.add_command(label="添加说话人", command=self.add_speaker_to_selection)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="复制", command=lambda: self.subtitle_text.event_generate("<<Copy>>"))
        self.context_menu.add_command(label="粘贴", command=lambda: self.subtitle_text.event_generate("<<Paste>>"))
        self.context_menu.add_command(label="剪切", command=lambda: self.subtitle_text.event_generate("<<Cut>>"))
        
        # 初始隐藏说话人选择区域
        self.speaker_selector_frame.pack_forget()
    
    def show_context_menu(self, event):
        """显示右键菜单"""
        self.context_menu.post(event.x_root, event.y_root)
    
    def toggle_speaker_mode(self):
        """切换说话人模式"""
        if self.speaker_mode.get():
            self.speaker_selector_frame.pack(fill="x", pady=5)
        else:
            self.speaker_selector_frame.pack_forget()
    
    def manage_speakers(self):
        """管理说话人列表（支持添加、删除、重命名）"""
        dialog = tk.Toplevel(self.root)
        dialog.title("管理说话人")
        dialog.geometry("320x340")
        dialog.transient(self.root)
        dialog.grab_set()
        
        speaker_listbox = tk.Listbox(dialog, width=30)
        speaker_listbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        for speaker in self.speakers:
            speaker_listbox.insert(tk.END, speaker)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        def add_speaker():
            speaker = simpledialog.askstring("添加说话人", "请输入说话人名称:")
            if speaker and speaker.strip():
                speaker = speaker.strip()
                if speaker not in self.speakers:
                    self.speakers.append(speaker)
                    speaker_listbox.insert(tk.END, speaker)
                    self.speaker_combo['values'] = self.speakers
                    self.current_speaker.set(speaker)
        
        def delete_speaker():
            selection = speaker_listbox.curselection()
            if selection:
                index = selection[0]
                speaker = speaker_listbox.get(index)
                if speaker in self.speakers:
                    self.speakers.remove(speaker)
                    speaker_listbox.delete(index)
                    self.speaker_combo['values'] = self.speakers
                    # 删除后自动选中第一个
                    if self.speakers:
                        self.current_speaker.set(self.speakers[0])
                    else:
                        self.current_speaker.set("")
        
        def rename_speaker():
            selection = speaker_listbox.curselection()
            if selection:
                index = selection[0]
                old_name = speaker_listbox.get(index)
                new_name = simpledialog.askstring("重命名说话人", f"将“{old_name}”重命名为：")
                if new_name and new_name.strip() and new_name not in self.speakers:
                    new_name = new_name.strip()
                    # 更新 speakers 列表
                    self.speakers[index] = new_name
                    # 更新所有字幕中的说话人字段
                    for sub in self.subtitles_with_speakers:
                        if sub.get('speaker') == old_name:
                            sub['speaker'] = new_name
                    # 更新特征库
                    if old_name in self.generator.speaker_embeddings:
                        self.generator.speaker_embeddings[new_name] = self.generator.speaker_embeddings.pop(old_name)
                    # 刷新列表和下拉框
                    speaker_listbox.delete(index)
                    speaker_listbox.insert(index, new_name)
                    self.speaker_combo['values'] = self.speakers
                    self.current_speaker.set(new_name)
                    # 刷新字幕区
                    self.update_subtitle_preview(self.subtitles_with_speakers)
                    self.save_subtitle_changes()
        
        ttk.Button(button_frame, text="添加", command=add_speaker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="删除", command=delete_speaker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="重命名", command=rename_speaker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side="right", padx=5)
    
    def adjust_speaker_count(self):
        """调整说话人数量的对话框"""
        if not self.subtitles_generated or not self.subtitles_with_speakers:
            messagebox.showerror("错误", "请先生成字幕并完成自动识别")
            return
        
        # 统计当前说话人数量
        current_speakers = set()
        for sub in self.subtitles_with_speakers:
            if 'speaker' in sub and sub['speaker']:
                current_speakers.add(sub['speaker'])
        current_count = len(current_speakers)
        
        # 输入目标数量
        target_count = simpledialog.askinteger(
            "调整说话人数量",
            f"当前识别到 {current_count} 个说话人，请输入目标数量:",
            minvalue=1,
            maxvalue=10,
            initialvalue=current_count
        )
        
        if target_count is not None and target_count != current_count:
            self.redistribute_speakers(target_count)
    
    def redistribute_speakers(self, target_count):
        """根据目标数量重新分配说话人"""
        if not self.subtitles_generated or not self.subtitles_with_speakers:
            messagebox.showerror("错误", "请先生成字幕并完成自动识别")
            return
        
        if target_count < 1:
            messagebox.showerror("错误", "目标数量必须至少为1")
            return
        
        current_speakers = list(set(
            speaker for sub in self.subtitles_with_speakers if 'speaker' in sub and sub['speaker'] for speaker in
            [sub['speaker']]))
        current_count = len(current_speakers)
        
        if current_count <= target_count:
            messagebox.showinfo("提示", f"当前说话人数量({current_count})已小于或等于目标数量({target_count})")
            return
        
        # 创建新的说话人名称列表
        new_speakers = [f"说话人{i + 1}" for i in range(target_count)]
        
        # 将新说话人添加到系统中
        for speaker in new_speakers:
            if speaker not in self.speakers:
                self.speakers.append(speaker)
        self.speaker_combo['values'] = self.speakers
        
        # 映射旧说话人到新说话人（按出现频率分配）
        speaker_frequency = {}
        for sub in self.subtitles_with_speakers:
            speaker = sub.get('speaker')
            if speaker:
                speaker_frequency[speaker] = speaker_frequency.get(speaker, 0) + 1
        
        # 按频率排序旧说话人
        sorted_old_speakers = sorted(speaker_frequency.keys(), key=lambda x: -speaker_frequency[x])
        
        # 创建映射表（前target_count个旧说话人保留原名，其余合并）
        speaker_mapping = {}
        for i, old_speaker in enumerate(sorted_old_speakers):
            if i < target_count:
                # 保留前target_count个说话人
                speaker_mapping[old_speaker] = old_speaker
                # 确保在新列表中
                if old_speaker not in new_speakers:
                    new_speakers.append(old_speaker)
            else:
                # 合并到最接近的新说话人
                mapped_index = i % target_count
                speaker_mapping[old_speaker] = new_speakers[mapped_index]
        
        # 应用映射表更新所有字幕
        for sub in self.subtitles_with_speakers:
            if 'speaker' in sub and sub['speaker'] in speaker_mapping:
                sub['speaker'] = speaker_mapping[sub['speaker']]
        
        # 更新说话人特征库
        new_embeddings = {}
        for old_speaker, embeddings in self.generator.speaker_embeddings.items():
            if old_speaker in speaker_mapping:
                new_speaker = speaker_mapping[old_speaker]
                if new_speaker not in new_embeddings:
                    new_embeddings[new_speaker] = []
                new_embeddings[new_speaker].extend(embeddings)
        self.generator.speaker_embeddings = new_embeddings
        
        # 更新界面
        self.update_subtitle_preview(self.subtitles_with_speakers)
        self.save_subtitle_changes()
        
        messagebox.showinfo("成功", f"已将说话人数量调整为 {target_count} 个")
    
    def add_speaker_to_selection(self):
        """在选中文本前添加说话人标签"""
        if not self.speaker_mode.get():
            return
        
        try:
            selected_text = self.subtitle_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            if not selected_text:
                return
            
            speaker = self.current_speaker.get()
            new_text = f"{speaker}: {selected_text}"
            
            self.subtitle_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.subtitle_text.insert(tk.SEL_FIRST, new_text)
            
            self.subtitle_text.tag_add(tk.SEL, tk.SEL_FIRST, f"{tk.SEL_FIRST}+{len(new_text)}c")
            self.subtitle_text.mark_set(tk.INSERT, f"{tk.SEL_FIRST}+{len(new_text)}c")
            self.subtitle_text.see(tk.INSERT)
        except tk.TclError:
            pass
    
    def apply_speaker_to_selection(self):
        """将当前说话人应用到选中的字幕"""
        if not self.speaker_mode.get():
            return
        
        try:
            current_pos = self.subtitle_text.index(tk.INSERT)
            start_pos = self.find_subtitle_start(current_pos)
            end_pos = self.find_subtitle_end(current_pos)
            
            if start_pos and end_pos:
                subtitle_text = self.subtitle_text.get(start_pos, end_pos)
                lines = subtitle_text.strip().split('\n')
                if len(lines) >= 3:
                    text_lines = lines[2:]
                    clean_text = []
                    for line in text_lines:
                        if re.match(r'^[^:]+: ', line):
                            line = line.split(': ', 1)[1]
                        clean_text.append(line)
                    
                    speaker = self.current_speaker.get()
                    speaker_line = f"{speaker}: {clean_text[0]}"
                    
                    new_subtitle = '\n'.join([lines[0], lines[1], speaker_line] + clean_text[1:])
                    
                    self.subtitle_text.delete(start_pos, end_pos)
                    self.subtitle_text.insert(start_pos, new_subtitle)
                    
                    self.subtitle_text.mark_set(tk.INSERT, f"{start_pos}+{len(new_subtitle)}c")
                    self.subtitle_text.see(tk.INSERT)
        except Exception as e:
            print(f"应用说话人时出错: {str(e)}")
    
    def apply_speaker_to_all(self):
        """将当前说话人应用到所有字幕"""
        if not self.speaker_mode.get():
            return
        
        if not messagebox.askyesno("确认", "确定要将当前说话人应用到所有字幕吗？"):
            return
        
        try:
            all_text = self.subtitle_text.get(1.0, tk.END)
            subtitle_blocks = re.split(r'\n\s*\n', all_text.strip())
            
            new_subtitles = []
            speaker = self.current_speaker.get()
            
            for block in subtitle_blocks:
                if not block.strip():
                    continue
                
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    new_subtitles.append(block)
                    continue
                
                text_lines = lines[2:]
                clean_text = []
                for line in text_lines:
                    if re.match(r'^[^:]+: ', line):
                        line = line.split(': ', 1)[1]