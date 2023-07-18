import cv2
import torch
import torchvision.transforms as transforms

def extract_keyframes(video_path, threshold=0.5):
    """Extract key frames from the video."""
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = num_frames // 2

    # Initialize a tensor to store the previous frame
    transform = transforms.ToTensor()
    _, frame = cap.read()
    prev_frame = transform(frame).unsqueeze(0)

    keyframes = []
    mid_frame = None
    for i in range(1, num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Save the middle frame
        if i == mid_frame_idx:
            mid_frame = frame

        # Convert the frame to a tensor
        curr_frame = transform(frame).unsqueeze(0)

        # Compute the absolute difference between the current frame and the previous frame
        diff = torch.abs(curr_frame - prev_frame).mean()

        # If the difference is above a certain threshold, save the frame as a keyframe
        if diff > threshold:
            keyframes.append(frame)

        prev_frame = curr_frame

    # If no keyframe found, use the middle frame as the keyframe
    if not keyframes and mid_frame is not None:
        keyframes.append(mid_frame)

    cap.release()
    return keyframes


if __name__ == "__main__":
    keyframes = extract_keyframes("./basketball.mp4")
    for frame in keyframes:
        cv2.imshow("frame", frame)
        cv2.waitKey(0)