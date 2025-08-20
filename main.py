import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Pothole Detection App",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Pothole Detection with YOLOv8")
st.markdown("Upload an image or video to detect potholes using trained YOLOv8 model")

@st.cache_resource
def load_model():
    # Load the YOLOv8 model with caching for better performance
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO('best.pt').to(device=device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'best.pt' file is in the same directory as this script")
        return None, None

def resize_image_for_detection(image, min_width=800, min_height=600):
    # Resize image if it's too small for better detection visibility
    width, height = image.size
    
    # Calculate if resizing is needed
    if width < min_width or height < min_height:
        # Calculate scaling factor to meet minimum dimensions
        scale_x = min_width / width if width < min_width else 1
        scale_y = min_height / height if height < min_height else 1
        scale_factor = max(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize using high-quality resampling
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image, scale_factor, True
    
    return image, 1.0, False

def process_image_with_detections(image, model, confidence_threshold):
    # Process image and return image with detections
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(opencv_image, conf=confidence_threshold)[0]
    
    # Draw bounding boxes and labels
    annotated_image = opencv_image.copy()
    detection_count = 0
    
    if results.boxes is not None:
        for box in results.boxes:
            detection_count += 1
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Adjust line thickness and font size based on image size
            line_thickness = max(2, int(min(image.size) / 400))
            font_scale = max(0.5, min(image.size) / 1200)
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
            
            # Draw label with background
            label_text = f"{label} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
            
            # Ensure label doesn't go outside image bounds
            label_y = max(text_height + 10, y1)
            cv2.rectangle(annotated_image, (x1, label_y - text_height - 10), (x1 + text_width, label_y), (0, 255, 0), -1)
            cv2.putText(annotated_image, label_text, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), line_thickness)
    
    # Convert back to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, detection_count

def check_ffmpeg_available():
    """Check if FFmpeg is available in the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

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

def process_video(video_path, model, confidence_threshold, progress_bar):
    # Process video and return path to processed video (fallback method)
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:  # Handle invalid fps
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure dimensions are even (required for some codecs)
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1
    
    # Create output video writer with web-compatible codec
    output_path = tempfile.mktemp(suffix='.mp4')
    
    # Use mp4v codec only (most compatible)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Could not create video writer")
        return None, 0
    
    frame_count = 0
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame if dimensions were adjusted
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
            
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
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Ensure label stays within frame
                label_y = max(text_height + 10, y1)
                cv2.rectangle(frame, (x1, label_y - text_height - 10), 
                            (min(x1 + text_width, width), label_y), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (x1, label_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        total_detections += frame_detections
        out.write(frame)
        
        # Update progress
        frame_count += 1
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
    
    cap.release()
    out.release()
    
    # Verify the output file was created and has content
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        st.error("Output video file was not created properly")
        return None, 0
    
    return output_path, total_detections

def main():
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Display model info
    st.success(f"✅ Model loaded successfully on {device}")
    
    # Check FFmpeg availability
    ffmpeg_available = check_ffmpeg_available()
    if ffmpeg_available:
        st.info("🎬 FFmpeg detected - Enhanced video compatibility enabled")
    else:
        st.warning("⚠️ FFmpeg not available - Using basic video processing")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: JPG, PNG, MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            # Process image
            st.subheader("📸 Image Processing")
            
            # Load original image
            original_image = Image.open(uploaded_file)
            original_size = original_image.size
            
            # Check if image needs resizing for better detection visibility
            resized_image, scale_factor, was_resized = resize_image_for_detection(original_image)
            
            # Display original image info
            st.info(f"📏 Original Image Size: {original_size[0]} × {original_size[1]} pixels")
            
            if was_resized:
                st.info(f"🔍 Image automatically resized to {resized_image.size[0]} × {resized_image.size[1]} pixels (scale: {scale_factor:.2f}x) for better detection visibility")
                
                with st.spinner("Processing images..."):
                    # Process both original and resized images
                    original_processed, original_count = process_image_with_detections(original_image, model, confidence_threshold)
                    resized_processed, resized_count = process_image_with_detections(resized_image, model, confidence_threshold)
                
                # Show results in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Size Result**")
                    st.image(original_processed, use_container_width=True)
                    st.write(f"Size: {original_size[0]} × {original_size[1]} pixels")
                    if original_count > 0:
                        st.success(f"🎯 Detected {original_count} pothole(s)")
                    else:
                        st.warning("⚠️ No potholes detected")
                
                with col2:
                    st.write("**Resized Version Result**")
                    st.image(resized_processed, use_container_width=True)
                    st.write(f"Size: {resized_image.size[0]} × {resized_image.size[1]} pixels")
                    if resized_count > 0:
                        st.success(f"🎯 Detected {resized_count} pothole(s)")
                    else:
                        st.warning("⚠️ No potholes detected")
                
                # Show comparison
                if resized_count > original_count:
                    st.info(f"💡 Resizing improved detection: {resized_count - original_count} additional pothole(s) detected!")
                elif original_count > resized_count:
                    st.info(f"💡 Original size performed better: {original_count - resized_count} more pothole(s) detected!")
                else:
                    st.info("💡 Both versions detected the same number of potholes")
                
                # Download buttons
                col1_dl, col2_dl = st.columns(2)
                
                with col1_dl:
                    # Download original size processed image
                    original_pil = Image.fromarray(original_processed)
                    img_buffer_orig = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    original_pil.save(img_buffer_orig.name, format='JPEG')
                    
                    with open(img_buffer_orig.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download Original Size",
                            data=f.read(),
                            file_name=f"original_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
                
                with col2_dl:
                    # Download resized processed image
                    resized_pil = Image.fromarray(resized_processed)
                    img_buffer_res = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    resized_pil.save(img_buffer_res.name, format='JPEG')
                    
                    with open(img_buffer_res.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download Resized Version",
                            data=f.read(),
                            file_name=f"resized_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
            
            else:
                # Image is already large enough, process normally
                with st.spinner("Processing image..."):
                    processed_image, detection_count = process_image_with_detections(original_image, model, confidence_threshold)
                
                # Show single result
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image**")
                    st.image(original_image, use_container_width=True)
                
                with col2:
                    st.write("**Processed Result**")
                    st.image(processed_image, use_container_width=True)
                
                # Display statistics
                if detection_count > 0:
                    st.success(f"🎯 Detected {detection_count} pothole(s)")
                else:
                    st.warning("⚠️ No potholes detected. Try adjusting the confidence threshold.")
                
                # Download button
                processed_pil = Image.fromarray(processed_image)
                img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                processed_pil.save(img_buffer.name, format='JPEG')
                
                with open(img_buffer.name, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=f.read(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
        
        elif file_type == 'video':
            # Process video
            st.subheader("🎥 Video Processing")
            
            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name
            
            # Display original video
            st.write("**Original Video**")
            st.video(uploaded_file)
            
            # Process video with progress bar
            st.write("**Processing Video...**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing video frames..."):
                if ffmpeg_available:
                    processed_video_path, total_detections = process_video_with_ffmpeg(
                        temp_video_path, model, confidence_threshold, progress_bar
                    )
                else:
                    processed_video_path, total_detections = process_video(
                        temp_video_path, model, confidence_threshold, progress_bar
                    )
            
            if processed_video_path and os.path.exists(processed_video_path):
                progress_bar.progress(1.0)
                status_text.success(f"✅ Video processing complete! Total detections: {total_detections}")
                
                # Display processed video
                st.write("**Processed Video**")
                
                # Read the processed video file
                try:
                    with open(processed_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    if len(video_bytes) == 0:
                        st.error("❌ Processed video file is empty")
                    else:
                        # Display the processed video
                        st.video(video_bytes)
                        
                        # Download button for processed video
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name=f"detected_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                        
                        # Show video info
                        st.info(f"📊 Processed video size: {len(video_bytes) / (1024*1024):.1f} MB")
                        
                except Exception as e:
                    st.error(f"❌ Error reading processed video: {str(e)}")
                
                # Cleanup temp files
                try:
                    os.unlink(temp_video_path)
                    os.unlink(processed_video_path)
                except:
                    pass
            else:
                st.error("❌ Error processing video. Please try again or use a different video format.")
    
    else:
        # Display instructions when no file is uploaded
        st.info("""
        👆 **Upload an image or video to get started!**
        
        **Supported formats:**
        - **Images:** JPG, JPEG, PNG, BMP, TIFF
        - **Videos:** MP4, AVI, MOV, MKV
        
        **Features:**
        - **Automatic Smart Resizing:** Small images are automatically resized for better detection
        - **Dual Output:** See both original and resized results when applicable
        - **Real-time Pothole Detection:** Powered by YOLOv8
        - **Enhanced Video Processing:** FFmpeg support for better web compatibility
        - **Adjustable Confidence:** Fine-tune detection sensitivity
        - **Download Results:** Save processed files
        - **Performance Comparison:** See which size performs better
        """)

if __name__ == "__main__":
    main()