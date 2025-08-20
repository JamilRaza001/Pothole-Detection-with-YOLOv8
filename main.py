import subprocess
import sys

def process_video_with_ffmpeg(video_path, model, confidence_threshold, progress_bar):
    """Process video and convert to web-compatible format using FFmpeg"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output paths
    temp_output = tempfile.mktemp(suffix='.avi')  # Use AVI for processing
    final_output = tempfile.mktemp(suffix='.mp4')  # Final MP4 output
    
    # Use XVID codec for processing (more reliable)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Could not create video writer")
        return None, 0
    
    frame_count = 0
    total_detections = 0
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference on frame
        results = model(frame, conf=confidence_threshold)[0]
        
        # Draw detections on frame
        frame_detections = 0
        if results.boxes is not None:
            for box in results.boxes:
                frame_detections += 1
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        total_detections += frame_detections
        out.write(frame)
        
        # Update progress
        frame_count += 1
        if total_frames > 0:
            progress = min(frame_count / total_frames * 0.8, 0.8)  # Reserve 20% for conversion
            progress_bar.progress(progress)
    
    cap.release()
    out.release()
    
    # Convert to web-compatible MP4 using FFmpeg
    try:
        progress_bar.progress(0.85)
        
        # FFmpeg command for web-optimized MP4
        cmd = [
            'ffmpeg', '-y', '-i', temp_output,
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'fast',   # Encoding speed preset
            '-crf', '23',        # Constant Rate Factor (quality)
            '-movflags', '+faststart',  # Optimize for web streaming
            '-pix_fmt', 'yuv420p',      # Pixel format for compatibility
            '-f', 'mp4',         # Force MP4 format
            final_output
        ]
        
        # Run FFmpeg conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            st.error(f"FFmpeg conversion failed: {result.stderr}")
            return temp_output, total_detections  # Return original if conversion fails
        
        progress_bar.progress(1.0)
        
        # Clean up temporary file
        try:
            os.unlink(temp_output)
        except:
            pass
        
        return final_output, total_detections
        
    except subprocess.TimeoutExpired:
        st.error("Video conversion timeout")
        return temp_output, total_detections
    except Exception as e:
        st.error(f"Error during video conversion: {str(e)}")
        return temp_output, total_detections

def check_ffmpeg_available():
    """Check if FFmpeg is available in the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False