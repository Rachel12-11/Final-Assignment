# Intelligent Subtitle Generator (with Speaker Labeling)

The Intelligent Subtitle Generator is a powerful and user-friendly tool designed specifically for video creators, media professionals, educators, and anyone who needs to add subtitles to videos. Leveraging advanced speech recognition technology and intelligent algorithms, it can quickly and accurately generate subtitles for various types of videos. Additionally, it supports speaker labeling, greatly enhancing the efficiency and quality of subtitle production.

## Core Features

### 1. Video File Processing
- Supports a variety of common video formats, such as MP4, MOV, AVI, MKV, etc., making it easy for users to import videos from different sources.
- Automatically detects the FFmpeg tool in the system. If not found, users can manually specify the FFmpeg path to ensure smooth video processing.

### 2. Subtitle Generation
- Integrates the Whisper speech recognition model, offering multiple model sizes (tiny, base, small, medium, large) to allow users to choose based on their needs and device performance, balancing accuracy and speed.
- Generates subtitles in SRT format for easy application in various video players, and also provides JSON format data for further analysis and processing.

### 3. Speaker Labeling
- Supports enabling speaker labeling, allowing users to manually manage the speaker list by adding, deleting, and renaming speakers.
- Offers automatic speaker recognition using clustering algorithms, with preset numbers of speakers and similarity thresholds for differentiation and labeling.
- Allows users to adjust the number of speakers dynamically; the system will automatically redistribute and ensure accurate speaker labels in the subtitles.

### 4. Subtitle Editing and Style Customization
- Provides an intuitive subtitle editing interface, enabling users to edit text directly, with typical operations like copy, paste, and cut.
- Supports adding speaker tags to selected or all subtitles for easy annotation and distinction.
- Allows customization of subtitle styles, including font, font size, and position (top, middle, bottom) to meet various visual requirements.

### 5. Video and Subtitle Merging
- Capable of embedding generated subtitles into the original video, producing a new video file with subtitles in various formats.

---
This tool aims to streamline the process of subtitle creation, making it faster and more accurate while offering flexible customization options to suit diverse needs.
