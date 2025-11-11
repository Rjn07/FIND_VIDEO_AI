from flask import Flask, render_template, request, jsonify, send_file
import torch
import open_clip
from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load CLIP model
print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

# Try to load SAM model
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = False
    print("Loading SAM model...")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("SAM model loaded successfully!")
except Exception as e:
    SAM_AVAILABLE = False
    print(f"SAM not available: {e}")


def extract_frames(video_path, frames_per_second=5):
    """Extract frames from video at a specified rate."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_ids = []
    frame_count = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = int(fps / frames_per_second)
    
    video_info = {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'frame_interval': frame_interval,
        'frames_per_second': frames_per_second
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
            timestamp = frame_count / fps
            frame_ids.append(timestamp)
        
        frame_count += 1

    cap.release()
    return frames, frame_ids, video_info


def get_clip_embeddings(frames, query, batch_size=32):
    """Compute CLIP embeddings for frames and text query."""
    text = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    all_image_features = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        image_tensors = torch.stack([preprocess(f) for f in batch_frames]).to(DEVICE)
        
        with torch.no_grad():
            batch_features = model.encode_image(image_tensors)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            all_image_features.append(batch_features.cpu())
    
    image_features = torch.cat(all_image_features, dim=0)
    return image_features, text_features.cpu()


def apply_sam_segmentation(image):
    """Apply SAM segmentation to get bounding boxes."""
    if not SAM_AVAILABLE:
        return []
    
    image_np = np.array(image)
    masks = mask_generator.generate(image_np)
    
    bboxes = []
    for mask in masks:
        bbox = mask['bbox']
        area = mask['area']
        if area > 1000:
            bboxes.append(bbox)
    
    return bboxes


def image_to_base64(image, bboxes=None):
    """Convert PIL image to base64 string with optional bounding boxes."""
    img_np = np.array(image)
    
    # Draw bounding boxes if available
    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(img_np, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    
    # Convert to JPEG
    img_pil = Image.fromarray(img_np)
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def format_timestamp(seconds):
    """Convert seconds to MM:SS.ms format"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        # Get form data
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        query = request.form.get('query', '')
        frames_per_second = int(request.form.get('fps', 5))
        top_k = int(request.form.get('top_k', 5))
        
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        # Save uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        # Process video
        frames, frame_times, video_info = extract_frames(video_path, frames_per_second)
        
        if len(frames) == 0:
            return jsonify({'error': 'No frames extracted from video'}), 400
        
        # Get embeddings and compute similarity
        image_features, text_features = get_clip_embeddings(frames, query)
        similarity = (image_features @ text_features.T).squeeze()
        
        # Get top k matches
        k = min(top_k, len(similarity))
        topk = similarity.topk(k)
        
        # Prepare results
        results = []
        for score, frame_idx in zip(topk.values, topk.indices):
            idx = frame_idx.item()
            timestamp = frame_times[idx]
            score_val = score.item()
            
            # Get frame and apply segmentation
            frame = frames[idx]
            bboxes = apply_sam_segmentation(frame) if SAM_AVAILABLE else []
            
            # Convert frame to base64
            frame_base64 = image_to_base64(frame, bboxes)
            
            results.append({
                'timestamp': timestamp,
                'formatted_time': format_timestamp(timestamp),
                'score': float(score_val),
                'image': frame_base64,
                'objects_detected': len(bboxes)
            })
        
        # Clean up
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'results': results,
            'video_info': {
                'duration': video_info['duration'],
                'fps': video_info['fps'],
                'total_frames': video_info['total_frames'],
                'frames_extracted': len(frames)
            },
            'sam_available': SAM_AVAILABLE
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run( host='0.0.0.0', port=5000, debug=True)